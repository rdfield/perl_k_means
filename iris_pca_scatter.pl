use Modern::Perl;
use Text::CSV qw(csv);
use Data::Dumper;
use lib '/home/mvine/ownCloud/perl_ml_relu';
use ML::Util qw(print_2d_array);
use ML::PCA;
use ML::KMeans;
use List::Util qw(zip);

use Chart::Scatter;

my $debug = 0;

sub save_chart
{
        my $chart = shift or die "Need a chart!";
        my $name = shift or die "Need a name!";
 
        my $ext = $chart->export_format;
 
        open(my $out, '>', "$name.$ext") or
                die "Cannot open '$name.$ext' for write: $!";
        binmode $out;
        print $out $chart->gd->$ext();
        close $out;
}

my $cluster_count = 3;
my $csv = Text::CSV->new();
my $fh;
open $fh, "<", "iris_truncated.csv";
my $data = $csv->fragment($fh, "col=1-4");
close $fh;
#my $headers = shift @$data;


my $pca = ML::PCA->new(#GPU => "ROCM", 
debug => $debug, 
threshold => 0.00001,
max_iterations => 150);
my $coords = $pca->project( $data, 2 );
print_2d_array("coords", $coords) if $debug;

my $kmeans = ML::KMeans->new();
my $clusters = $kmeans->clusterise(data => $coords, clusters => $cluster_count);

my $centroids = $kmeans->centroids();
print_2d_array("final centroids", $centroids) if $debug;

my @graph_data;
my @legend_labels;
my $my_graph = Chart::Scatter->new( 800, 600 );
foreach (zip $coords, $clusters) {
   my ($coord, $cluster) = @$_;
   #say join(",",@$coord) . ",$cluster";
   push @{$graph_data[0]}, $coord->[0];
   foreach my $i (1 .. $cluster_count ) {
      if (($i - 1) == $cluster ) {
         push @{$graph_data[$i]}, $coord->[1];
      } else {
         push @{$graph_data[$i]}, undef;
      }
   }
}
foreach my $i (0 .. $#{$graph_data[0]}) {
   push @{$graph_data[$cluster_count + 1]}, undef;
}
foreach my $i (0 .. $cluster_count - 1) {
   push @legend_labels, "Cluster $i";
}
foreach my $c (@$centroids) {
   push @{$graph_data[0]}, $c->[0];
   foreach my $i (1 .. $cluster_count) {
      push @{$graph_data[$i]}, undef;
   }
   push @{$graph_data[$cluster_count + 1]}, $c->[1];
}
push @legend_labels, "Centroids";

$my_graph->set(legend_labels => \@legend_labels,
               skip_x_ticks => 10,
               f_x_tick => sub { sprintf("%.3f", $_[0]) },
               max_x_ticks => 5);
           
foreach my $g (@graph_data) {
   $my_graph->add_dataset($g);
}
my $file_name = "cluster_plot_$$.png";
$my_graph->png($file_name);
system("feh $file_name &");
