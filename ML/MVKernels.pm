use Modern::Perl;
package ML::MVKernels;
use Math::Matrix;
use Math::Random;
use Data::Dumper;
use File::Slurp;
use List::Util qw/shuffle/;
use Time::HiRes qw(gettimeofday tv_interval);
use Cwd qw(abs_path);
use JSON;

my $gpuif;
sub import {
   my $self = shift;
   if (!scalar(@_) or $_[0] eq "CUDA") {
      require ML::MVCUDA;
      $gpuif = ML::MVCUDA->new();
   } else {
      require ML::MVROCM;
      $gpuif = ML::MVROCM->new();
   }
}

sub new {
   my $class = shift;
   my %params = @_;
   my $self = {};
   if (defined($params{debug}) and $params{debug} == 1) {
      $self->{debug} = 1;
   } else {
      $self->{debug} = 0;
   }
   return bless $self, $class;
}


my %loss_functions = (
                       "quadratic" => 1,
                       "cel" => 2
                     );

my %network_init_params;

sub create_network {
   my $self = shift;
   my ($sizes, @options) = @_;
   $network_init_params{ sizes } = $sizes;
   my %params = @options;
   $params{batch_size} ||= 10;
   my $scale_factor = 1;
   if (defined($params{weight_init}) and $params{weight_init} eq "scaled") {
      $scale_factor = sqrt($params{batch_size});
   }
   foreach my $i (0 .. ($#{$sizes} - 1)) {
      # "input size" = $sizes->[$i]
      # "output size" = $sizes->[$i + 1]
      if (!defined($params{ bias }->[$i])) {
         # this will be an "output size" x 1 array
         $params{bias}->[$i] = Math::Matrix->new([random_normal($sizes->[$i + 1])])->transpose()->as_array();
      }
      if (!defined($params{ weights }->[$i])) {
         # this will be an "output size" x "input size" array
         my $iw = [];
         foreach my $s (1 .. $sizes->[$i + 1]) {
            push @$iw, [map { $_ / $scale_factor } random_normal( $sizes->[$i] ) ];
         }
         $params{weights}->[$i] = $iw;
      }
      return 0 unless $gpuif->c_add_node($sizes->[$i], $sizes->[$i + 1], $params{bias}->[$i], $params{weights}->[$i], $params{batch_size});
   }
   # reserve RAM for initial input
   $gpuif->c_reset_derivatives();
   if ($self->{debug} == 1 or ($params{debug} and $params{debug} == 1)) {
      $gpuif->c_set_debug_on();
   } else {
      $gpuif->c_set_debug_off();
   }
   if ($params{loss} =~ /(CrossEntropy|CrossEntropyLoss|CEL)/i) {
      $gpuif->c_set_loss($loss_functions{cel});
      $network_init_params{loss} = "cel";
   } else {
      $gpuif->c_set_loss($loss_functions{quadratic});
      $network_init_params{loss} = "mse";
   }
   return 0 unless $gpuif->c_reserve_input_memory($sizes->[0], $sizes->[-1], $params{batch_size});
   return 1;
}

sub print_network {
   $gpuif->c_print_list();
}

sub feedforward {
   my $xy = shift;
   # convert input to C, put into already reserved memory
   my (@x, @y);
   my $elements = 0;
   foreach my $input (@$xy) {
      push @x, $input->[0];
      push @y, $input->[1];
      $elements++;
   }
   return unless $gpuif->c_load_input(\@x, $elements);
   return unless $gpuif->c_load_target(\@y);
   # run the forward pass of the network
   $gpuif->c_run_feed_forward();
   #my $last_activated_output = [];
   #get_last_activated_output($last_activated_output) ;
   #Math::Matrix->new($last_activated_output)->print("Last activated output");
}

sub validation_feedforward {
   my $xy = shift;
   # convert input to C, put into already reserved memory
   my (@x, @y);
   my $elements = 0;
   foreach my $input (@$xy) {
      push @x, $input->[0];
      push @y, $input->[1];
      $elements++;
   }
   return unless $gpuif->c_load_input(\@x, $elements);
   return unless $gpuif->c_load_target(\@y);
   # run the forward pass of the network
   $gpuif->c_run_feed_forward();
   my $last_activated_output = [];
   $gpuif->c_get_last_activated_output($last_activated_output) ;
   return $last_activated_output;
}

sub calculate_loss {
   $gpuif->c_calculate_cost_derivative();
}

sub backprop {
   $gpuif->c_run_backpropagation();
}

sub update_weights {
  # run at end of mini batch
  # don't forget to zero the derivatives!
  my %params = @_;
  $params{batch_size} ||= 10;
  $params{learning_rate} ||= 3;
  $params{decay} ||= 1;
  $gpuif->c_run_update_weights_and_biases( $params{learning_rate} / $params{batch_size}, $params{decay} );
  $gpuif->c_reset_derivatives();
}

sub update_mini_batch {
   my ($mb, $eta, $j, $ctr, $decay) = @_;
   feedforward($mb);
   calculate_loss();
   backprop();
   update_weights( batch_size => scalar(@$mb), learning_rate => $eta, decay => $decay );
}

sub argmax {
   my $arr = shift;
   my @max;
   foreach my $i (0 .. $#{$arr->[0]}) {
      my $max = $arr->[0][$i];
      $max[$i] = 0;
      my $idx = 0;
      foreach my $j ( 0 .. $#$arr) {
         if ($arr->[$j][$i] > $max) {
            $max = $arr->[$j][$i];
            $max[$i] = $j;
         }
      }
   }
   return \@max;
}

my @evaluation_batches;
my @evaluation_targets;
my @testing_batches;
my @testing_targets;


sub total_cost {
   my $data = shift;
   my $targets = shift;
   my $lambda = shift;
   my $cost = shift;
   my $accuracy = shift;
   my $data_len = 0;
   my $total_cost = 0;
   my $successes = 0;
   foreach my $i (0 .. $#$data) {
      my $calc;
      if ($accuracy) {
         $calc = validation_feedforward($data->[$i]);
      } else {
         feedforward($data->[$i]);
      }
      if ($cost) {
         $total_cost += calculate_cost();
         $data_len += scalar(@{$data->[$i]});
      }
      if ($accuracy) {
         my $max_calc_idx = argmax($calc);
         my $max_target_idx = argmax($targets->[$i]);
         foreach my $i (0 .. $#$max_calc_idx) {
            $successes++ if  $max_calc_idx->[$i] == $max_target_idx->[$i];
         }
      }
   }
   if ($cost) {
      $total_cost += calculate_weights_cost();
      $total_cost /= $data_len;
   }
   return $total_cost, $successes;
}

sub _cache_eval_data {
   my $mini_batch_size = shift;
   my $data = shift;
   my $data_cache = shift;
   my $target_cache = shift;
   my $k = 0;
   my $data_size = scalar(@$data);
   while (($k * $mini_batch_size) < $data_size) {
      push @$data_cache, [@$data[($k * $mini_batch_size) .. ((($k + 1) * $mini_batch_size) - 1)]];
      my @y;
      foreach my $i (0 .. ($mini_batch_size - 1)) {
         push @y, $data_cache->[-1][$i][1];
      }
      push @$target_cache, Math::Matrix->new(\@y)->transpose->as_array();
      $k++;
   }
}

sub SGD {
   my $self = shift;
   my $training_data = shift;
   my $epochs = shift;
   my $mini_batch_size = shift;
   my $eta = shift;
   my %params = @_;
   $params{load_from_file} ||= 0;
   my $test_data = $params{test_data};
   my $n = scalar(@$training_data);
   my $n_test;
   if ($test_data) {
      $n_test = scalar(@$test_data);
      _cache_eval_data($mini_batch_size, $test_data, \@testing_batches, \@testing_targets);
   }
   my $evaluation_data = $params{evaluation_data};
   my $n_eval;
   if ($evaluation_data) {
      $n_eval = scalar(@$evaluation_data);
      _cache_eval_data($mini_batch_size, $evaluation_data, \@evaluation_batches, \@evaluation_targets);
   }
   $params{lambda} ||= 0;
   my $decay = 1 - $eta * ( $params{lambda} / $n_test );
   my (@evaluation_cost, @evaluation_accuracy, @training_cost, @training_accuracy);
   foreach my $j (1 .. $epochs) {
      my $start = [ gettimeofday ];
      my @training_data = shuffle(@$training_data);
      my @mini_batches;  
      my $k = 0;
      while (scalar(@training_data)) {
         push @mini_batches, [ splice @training_data, 0 , $mini_batch_size ];
      } 
      my $ctr = 1;
      foreach my $mb (@mini_batches) {
         update_mini_batch($mb, $eta, $j, $ctr, $decay);
         $ctr++;
      } 
      say "Epoch $j training complete";
      if (defined($params{monitor_training_cost}) or defined($params{monitor_training_accuracy})) {
         my ($cost,$accuracy) = total_cost( \@testing_batches, \@testing_targets, $params{ lambda }, $params{monitor_training_cost}, $params{monitor_training_accuracy} );
         push @training_cost, $cost;
         say "Cost on training data $cost" if defined($params{monitor_training_cost});
         say "Accuracy on training data $accuracy / $n_test" if defined($params{monitor_training_accuracy})
      }
      if (defined($params{monitor_evaluation_cost}) or defined($params{monitory_evaluation_accuracy})) {
         my ($cost,$accuracy) = total_cost( \@evaluation_batches, \@evaluation_targets, $params{ lambda },  $params{monitor_evaluation_cost}, $params{monitor_evaluation_accuracy} );
         push @evaluation_cost, $cost;
         say "Cost on evaluation data $cost" if defined($params{monitor_evaluation_cost});
         say "Accuracy on evaluation data $accuracy / $n_eval" if defined($params{monitor_evaluation_accuracy})
      }
      say "epoch time = " . tv_interval($start , [gettimeofday]);
   }
   return { evaluation_cost => \@evaluation_cost,
            evaluation_accuracy => \@evaluation_accuracy,
            training_cost => \@training_cost,
            training_accuracy => \@training_accuracy };

}

sub save_network {
   my $self = shift;
   my $filename = shift;
   my $data = { loss => $network_init_params{loss}, sizes => $network_init_params{sizes} };
   foreach my $i (0 .. ($#{$network_init_params{sizes}} - 1)) {
      my $W = [];
      $gpuif->c_get_weights($W, $i);
      push @{$data->{weights}}, $W;
      my $B = [];
      $gpuif->c_get_biases($B, $i);
      push @{$data->{bias}}, $B;
   }
   open(FILE, ">", $filename);
   print FILE to_json($data);
   close FILE;
}   
    
sub load_network {
   my $self = shift;
   my $filename = shift;
   my $data = from_json(scalar(read_file($filename)));
   $self->create_network(delete $data->{sizes}, %$data);
}   
    
sub mnist_batch_guess {
# use if the data supplied is in the same format as the mnist batches
# otherwise use mnist_image_guess
   my $self = shift;
   my $data = shift;
   my $calc = validation_feedforward($data);
   return argmax($calc);
}   
    
sub mnist_image_guess {
   my $self = shift;
   my $data = shift;
   # expecting an array of 784 pixel values scaled to the 0-1 range
   $gpuif->c_set_debug_on();
   my @batch;
   $batch[0]->[0] = $data;
   $batch[0]->[1] = [(0) x 10]; # validation expects to see a target array, but it isn't needed, so just make it zeros.
   my $calc = validation_feedforward(\@batch);
   return argmax($calc);
}

sub calculate_covariance {
   my $self = shift;
   if ($self->{debug} == 1) {
      $gpuif->c_set_debug_on();
   }
   return $gpuif->c_calculate_covariance(@_);
}

sub eigenvectors {
   my $self = shift;
   my $pQ = shift;
   my $epsilon = shift;
   my $max_iterations = shift;
   if ($self->{debug} == 1) {
      $gpuif->c_set_debug_on();
   }
   $gpuif->c_eigenvectors($pQ, $epsilon, $max_iterations);
   return $pQ;
}

sub project_results {
   my $self = shift;
   my $columns = shift;
   my $projection = [];
   if ($self->{debug} == 1) {
      $gpuif->c_set_debug_on();
   }
   $gpuif->c_project_results($columns, $projection);
   return $projection;
}


1;
