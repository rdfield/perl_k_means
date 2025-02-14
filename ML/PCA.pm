use Modern::Perl;

package ML::PCA;

use List::Util qw(zip);
use Storable qw(dclone);
use ML::Util qw(transpose print_2d_array add_2_arrays diagonal_matrix matmul);
use lib '/home/mvine/ownCloud/perl_ml_relu';
use ML::MVKernels;

use Data::Dumper;

# from https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/GramSchmidt.pdf
# note that this version of the algorithm divides by || u || whereas others 
# divide by || u ||^2

sub project {
   my $self = shift;
   my $A = shift;
say "PCA project, A has " . scalar(@$A) . " rows, and there are " . scalar(@{$A->[0]}) . " columns in row 0" if $self->{debug};
   my $k = shift;
   $k ||= scalar(@{$A->[0]}); # if number of features, "k", isn't supplied, return all features
   $self->{cov} = [];
   $self->{Kernel}->calculate_covariance($A, $self->{cov});  # $A is the array ref to the original data, $self->{cov} will be populated
                                              # with the covariance data.  The C function will have the standardised & scaled
                                              # version of $A and also a copy of the covariant matrix (both in C & device 
                                              # structures, but not Perl)
                                              # Note that the device and host C arrays holding covariance data will be overwritten
                                              # by the eigenvalues when $Kernel->eigenvectors is called
# this algorithm from https://stats.stackexchange.com/questions/20643/finding-matrix-eigenvectors-using-qr-decomposition

   my $pQ = diagonal_matrix(1, $#{$self->{cov}});
   say "calulating eigenvectors " . localtime() if $self->{debug};
# note about sorting out the sign of each eigenvector
# https://stackoverflow.com/questions/17998228/sign-of-eigenvectors-change-depending-on-specification-of-the-symmetric-argument
   $self->{Kernel}->eigenvectors($pQ, $self->{epsilon}, $self->{max_iterations});
   say "eigenvectors complete " . localtime() if $self->{debug};
   print_2d_array("eigenvectors", $pQ) if $self->{debug};
=pod
   my $results = [];
   foreach my $r (@$pQ) {
      my @data;
      foreach my $i (0 .. ($k - 1)) {
         push @data, $r->[$i];
      }   
      push @$results, \@data;
   }
   say "eigenvectors sorted, and trimmed, starting projection " . localtime() if $self->{debug};
   print_2d_array("eigenvectors", $results) if $self->{debug};
=cut
   $self->{eigenvectors} = $pQ;
   my $projection = $self->{Kernel}->project_results($k); # Z and the eigenvectors will already be on the device
   $self->{projection} = $projection;
   print_2d_array("projection", $projection) if $self->{debug};
   return $projection;
}

sub new {
   my $class = shift;
   my %params = @_;
   my $self = {};
   if (defined($params{debug}) and $params{debug} =~ /^\d+$/ and $params{debug} > 0) {
      $self->{debug} = 1;
   } else {
      $self->{debug} = 0;
   }
   $self->{epsilon} = $params{threshold};
   $self->{epsilon} ||= 0.00001;
   $self->{epsilon} = 0.00001 unless $self->{epsilon} =~ /^\d+\.?\d+?$/;
   $self->{max_iterations} = $params{max_iterations};
   $self->{max_iterations} ||= 10;
   $self->{max_iterations} = 10 unless $self->{max_iterations} =~ /^\d+$/;
   require ML::MVKernels;
   if (!defined($params{GPU} ) or $params{GPU} ne "ROCM") {
      $params{GPU} = "CUDA";
   }
   ML::MVKernels->import($params{GPU});
   $self->{Kernel} = ML::MVKernels->new( debug => $self->{debug});
   bless $self, $class;
}

sub covariance {
   my $self = shift;
   return $self->{cov};
}
  
sub cumulative_explained_variance {
   my $self = shift;
   my $e_sum = 0;
   foreach my $s (@{$self->{eigenvector_sums}}) {
      $e_sum += $s->{value};
   }
   my $running_total = 0;
   return [ map { $running_total += $_->{value} / $e_sum; $running_total } @{$self->{eigenvector_sums}} ];
}
   
1;
