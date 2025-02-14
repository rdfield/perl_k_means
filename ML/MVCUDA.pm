use Modern::Perl;
package ML::MVCUDA;
use Math::Matrix;
use Math::Random;
use Data::Dumper;
use File::Slurp;
use List::Util qw/shuffle/;
use Time::HiRes qw(gettimeofday tv_interval);
use Cwd qw(abs_path);
use JSON;

my $code;

sub new {
   my $class = shift;
   my $self = {};
   return bless $self, $class;
}


use Inline CPP => Config =>
        BUILD_NOISY => 0,
        force_build => 0,
        clean_after_build => 1,
        warnings => 0,
        INC => "-I" . abs_path(substr(__FILE__,0,-1*(length("/ML/MVCUDA.pm"))) . "/inc")  . " -I" . abs_path("./inc") . " ",
        LIBS => "-L" . abs_path(substr(__FILE__,0,-1*(length("/ML/MVCUDA.pm"))) . "/lib") . " -L" . abs_path("./lib") . " -lCUDAKernel "
;

use Inline CPP => abs_path(substr(__FILE__,0,-1*(length("/MVCUDA.pm")))). "/MVKernels.c";

sub c_add_node {
   my $self = shift;
   return add_node(@_);
}

sub c_reset_derivatives {
   my $self = shift;
   return reset_derivatives();
}

sub c_set_debug_on {
   my $self = shift;
   set_debug_on();
}

sub c_set_debug_off {
   my $self = shift;
   set_debug_off();
}

sub c_set_loss {
   my $self = shift;
   return set_loss(@_);
}

sub c_reserve_input_memory {
   my $self = shift;
   return reserve_input_memory(@_);
}

sub c_print_list {
   my $self = shift;
   print_list();
}

sub c_load_input {
   my $self = shift;
   return load_input(@_);
}

sub c_load_target {
   my $self = shift;
   return load_target(@_);
}

sub c_run_feed_forward {
   my $self = shift;
   return run_feed_forward();
}

sub c_get_last_activated_output {
   my $self = shift;
   return get_last_activated_output(@_);
}

sub c_calculate_cost_derivative {
   my $self = shift;
   calculate_cost_derivative();
}

sub c_calculate_cost {
   my $self = shift;
   calculate_cost();
}

sub c_calculate_weights_cost {
   my $self = shift;
   calculate_weights_cost();
}
sub c_run_backpropagation {
   my $self = shift;
   run_backpropagation();
}

sub c_run_update_weights_and_biases {
   my $self = shift;
   return run_update_weights_and_biases( @_ );
}

sub c_get_weights {
   my $self = shift;
   return get_weights(@_);
}

sub c_get_biases {
   my $self = shift;
   return get_biases(@_);
}

sub c_calculate_covariance {
   my $self = shift;
   return calculate_covariance(@_);
}

sub c_eigenvectors {
   my $self = shift;
   return cuda_eigenvectors(@_);
}

sub c_project_results {
   my $self = shift;
   return cuda_project_results(@_);
}

1;
