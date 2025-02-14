package ML::KMeans;

use Modern::Perl;

use List::Util qw(zip);
use Data::Dumper;
use Text::CSV qw(csv);
use Cwd qw(abs_path);


my $code;
BEGIN {
   $code = <<'EOCODE';
// This section is boilerplace code to move data from Perl -> C and back again

#define HAVE_PERL_VERSION(R, V, S) \
    (PERL_REVISION > (R) || (PERL_REVISION == (R) && (PERL_VERSION > (V) || (PERL_VERSION == (V) && (PERL_SUBVERSION >= (S))))))

#define sv_setrv(s, r)  S_sv_setrv(aTHX_ s, r)

static void S_sv_setrv(pTHX_ SV *sv, SV *rv)
{
  sv_setiv(sv, (IV)rv);
#if !HAVE_PERL_VERSION(5, 24, 0)
  SvIOK_off(sv);
#endif
  SvROK_on(sv);
}

int is_array_ref(
        SV *array,
        size_t *array_sz
);
int array_numelts_2D(
        SV *array,
        size_t *_Nd1,
        size_t **_Nd2
);
int array_of_unsigned_int_into_AV(
        size_t *src,
        size_t src_sz,
        SV *dst
);
int array_of_int_into_AV(
        int *src,
        size_t src_sz,
        SV *dst
);

int is_array_ref(
        SV *array,
        size_t *array_sz
){
        if( ! SvROK(array) ){ fprintf(stderr, "is_array_ref() : warning, input '%p' is not a reference.\n", array); return 0; }
        if( SvTYPE(SvRV(array)) != SVt_PVAV ){ fprintf(stderr, "is_array_ref() : warning, input ref '%p' is not an ARRAY reference.\n", array); return 0; }
        // it's an array, cast it to AV to get its len via av_len();
        // yes, av_len needs to be bumped up
        int asz = 1+av_len((AV *)SvRV(array));
        if( asz < 0 ){ fprintf(stderr, "is_array_ref() : error, input array ref '%p' has negative size!\n", array); return 0; }
        *array_sz = (size_t )asz;
        return 1; // success, it is an array and size returned by ref, above
}

#define array_numelts_1D(A,B) (!is_array_ref(A,B))


#define array_numelts_1D(A,B) (!is_array_ref(A,B))

int array_numelts_2D(
        SV *array,
        size_t *_Nd1,
        size_t **_Nd2
){
        size_t anN, anN2, *Nd2 = NULL;

        if( ! is_array_ref(array, &anN) ){
           fprintf(stderr, "is_array_ref_2D() : error, call to is_array_ref() has failed for array '%p'.\n", array);
           return 1;
        }

        if( *_Nd2 == NULL ){
           if( (Nd2=(size_t *)malloc(anN*sizeof(size_t))) == NULL ){
               fprintf(stderr, "array_numelts_2D() : error, failed to allocate %zu bytes for %zu items for Nd2.\n", anN*sizeof(size_t), anN);
               return 1;
           }
        } else Nd2 = *_Nd2;
        AV *anAV = (AV *)SvRV(array);
        size_t *pNd2 = &(Nd2[0]);
        for(size_t i=0;i<anN;i++,pNd2++){
           SV *subarray = *av_fetch(anAV, i, FALSE);
           if( ! is_array_ref(subarray, &anN2) ){
              fprintf(stderr, "is_array_ref_2D() : error, call to is_array_ref() has failed for [%p][%p], item %zu.\n", array, subarray, i);
              if(*_Nd2==NULL) free(Nd2);
              return 1;
           }
           *pNd2 = anN2;
        }
        if( *_Nd2 == NULL ) *_Nd2 = Nd2;
        *_Nd1 = anN;
        return 0; // success
}

int array_of_int_into_AV(
        int *src,
        size_t src_sz,
        SV *dst
){
        size_t dst_sz;
        if( ! is_array_ref(dst, &dst_sz) ){ fprintf(stderr, "array_of_int_into_AV() : error, call to is_array_ref() has failed.\n"); return 1; }
        AV *dstAV = (AV *)SvRV(dst);
        for(size_t i=0;i<src_sz;i++){
                av_push(dstAV, newSViv(src[i]));
        }
        return 0; // success
}
// end of Perl -> C -> Perl section

void print_2D_array(float *foo, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%+.5f\t", foo[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

float  *host_centroids;
float  *host_centroids_work_area;
float  *host_data;
float  *host_distances;
size_t *host_cluster_map;
size_t CH, CW, DH, DW;
size_t *host_cluster_point_count;

int get_me_in_the_mood(SV *perl_centroids, SV *perl_data) {
   AV *av;
   float *pd;
   size_t *CWs = NULL, 
          *DWs = NULL, 
          i,j,insize;
   SV *subav, *subsubav, **ssubav;
   if( array_numelts_2D(perl_centroids, &CH, &CWs) ){
       fprintf(stderr, "initialise_me_freddo() : error, call to array_numelts_2D() has failed for input matrix centroids.\n");
       return 1;
   }
   if( array_numelts_2D(perl_data, &DH, &DWs) ){
       fprintf(stderr, "initialise_me_freddo() : error, call to array_numelts_2D() has failed for input matrix data.\n");
       return 1;
   }
   CW = CWs[0];
   DW = DWs[0];

   if( (host_centroids=(float *)malloc(CW*CH*sizeof(float))) == NULL ){
      fprintf(stderr, "initialise_me_freddo() : error, failed to allocate %zu bytes for %zu items for host_centroid.\n", CH*CW*sizeof(float), CH*CW);
      return 1;
   }
   if( (host_centroids_work_area=(float *)malloc(CW*CH*sizeof(float))) == NULL ){
      fprintf(stderr, "initialise_me_freddo() : error, failed to allocate %zu bytes for %zu items for host_centroid_work_area.\n", CH*CW*sizeof(float), CH*CW);
      return 1;
   }
   if( (host_cluster_point_count=(size_t *)malloc(CH*sizeof(size_t))) == NULL ){
      fprintf(stderr, "initialise_me_freddo() : error, failed to allocate %zu bytes for %zu items for host_cluster_point_count.\n", CH*sizeof(float), CH);
      return 1;
   }
   if( (host_data=(float *)malloc(DW*DH*sizeof(float))) == NULL ){
      fprintf(stderr, "initialise_me_freddo() : error, failed to allocate %zu bytes for %zu items for host_data.\n", DH*DW*sizeof(float), DH*DW);
      return 1;
   }
   if( (host_distances=(float *)malloc(DH*CH*sizeof(float))) == NULL ){ // 1 distance per centroid per data row
      fprintf(stderr, "initialise_me_freddo() : error, failed to allocate %zu bytes for %zu items for host_distances.\n", DH*CW*sizeof(float), DH*CW);
      return 1;
   }
   if( (host_cluster_map=(size_t *)malloc(DH*1*sizeof(size_t))) == NULL ){
      fprintf(stderr, "initialise_me_freddo() : error, failed to allocate %zu bytes for %zu items for host_cluster_map.\n", DH*1*sizeof(int), DH*CW);
      return 1;
   }

   pd = &(host_centroids[0]);
   av = (AV *)SvRV(perl_centroids);

   for(i=0;i<CH;i++){ // for each row
       subav = *av_fetch(av, i, FALSE);
       for(j=0;j<CW;j++){ // for the cols of that row
          subsubav = *av_fetch((AV *)SvRV(subav), j, FALSE);
          *pd = SvNV(subsubav);
          pd++;
       }
   }

   pd = &(host_data[0]);
   av = (AV *)SvRV(perl_data);

   for(i=0;i<DH;i++){ // for each row
       subav = *av_fetch(av, i, FALSE);
       for(j=0;j<DW;j++){ // for the cols of that row
          subsubav = *av_fetch((AV *)SvRV(subav), j, FALSE);
          *pd = SvNV(subsubav);
          pd++;
       }
   }
   return 0;
}

int bring_me_closer() {
   for (size_t i = 0; i < (CH * CW); i++) {
      host_centroids_work_area[i] = 0;
   }
   for (size_t i = 0; i < CH ; i++) {
      host_cluster_point_count[i] = 0;
   }
   for (size_t i = 0; i < DH; i++) {
      host_cluster_point_count[ host_cluster_map[ i ] ]++;
      for (size_t j = 0; j < DW; j++ ) {
         host_centroids_work_area[ host_cluster_map[ i ] * CW + j ] += host_data[ i * DW + j ];
      }
   }
   for (size_t i = 0; i < CH; i++) {
      for (size_t j = 0; j < CW; j++ ) {
         host_centroids[ i * CW + j] = host_centroids_work_area[ i * CW + j ] / host_cluster_point_count[ i ]; 
      }
   }
   return 0;
}

int are_we_there_yet() {
   size_t changes = 0; 
   for (size_t i = 0; i < DH; i++) { // for each data item
      for (size_t j = 0; j < CH; j++) { // for each centroid
         float distance = 0;
         for (size_t k = 0; k < CW; k++) { // for each dimension
            distance += powf( host_data[ DW * i + k] - host_centroids[ CW * j + k] ,2 )  ;
         }
         distance = sqrt( distance );
         host_distances[ i * CH + j ] = distance;
      }
      size_t minidx = 0;
      float min = host_distances[i * CH];
      for (size_t m = 0; m < CH; m++) {
         if (host_distances[i * CH + m] < min) {
            min = host_distances[i * CH + m];
            minidx = m;
         }
      }
      if (host_cluster_map[i] != minidx) {
         changes++;
         host_cluster_map[i] = minidx;
      }
   }
   return changes;
}

int take_me_home(SV *perl_R) {
   AV *av, *av2;
   size_t *pd;
   size_t j,RH,RW, asz;

   if( is_array_ref(perl_R, &asz) ){
            av = (AV *)SvRV(perl_R);
            if( asz > 0 ){
               av_clear(av);
            }
   } else if( SvROK(perl_R) ){
            av = newAV();
            // LeoNerd's suggestion:
            sv_setrv(SvRV(perl_R), (SV *)av);
   } else {
            av = newAV();
            // LeoNerd's suggestion:
            sv_setrv(perl_R, (SV *)av);
   }

   RH = DH;
   RW = 1;

   pd = &(host_cluster_map[0]);
   av = (AV *)SvRV(perl_R);
   for(int i=0;i<RH;i++){ // for each row
      av_push(av, newSVnv(*pd));
      pd++;
   }
   return 0;
}

int get_centroids(SV *perl_R) {
   AV *av, *av2;
   float *pd;
   size_t j,RH,RW, asz;
   
   if( is_array_ref(perl_R, &asz) ){
            av = (AV *)SvRV(perl_R);
            if( asz > 0 ){
               av_clear(av);
            }
   } else if( SvROK(perl_R) ){
            av = newAV();
            // LeoNerd's suggestion:
            sv_setrv(SvRV(perl_R), (SV *)av);
   } else {
            av = newAV();
            // LeoNerd's suggestion:
            sv_setrv(perl_R, (SV *)av);
   } 

   RH = CH;
   RW = CW;

   pd = &(host_centroids[0]);
   av = (AV *)SvRV(perl_R);
   for(int i=0;i<RH;i++){ // for each row
      av2 = newAV();
      av_extend(av2, RW);
      av_push(av, newRV_noinc((SV *)av2));
      for(j=0;j<RW;j++) {
         av_store(av2, j, newSVnv(*pd));
         pd++;
      }
   }
   return 0;
}
int clean_me_up_im_dirty() {
   free(host_centroids);
   free(host_centroids_work_area);
   free(host_data);
   free(host_distances);
   free(host_cluster_map);
   free(host_cluster_point_count);
   return 0;   
}
EOCODE


};
sub new {
   my $class = shift;
   return bless {}, $class;
}

sub update_clusters {
   my %args = @_;
   my $changes = 0;
   foreach my $d (@{$args{data}}) {
      my @d;
      foreach my $i (@{$args{centroids}}) {
         my $dist = 0;
         foreach (zip $i, $d->{$args{coords_key}}) {
            my ($centroid, $coord) = @$_;
            $dist += ($centroid - $coord)**2;
         }
         $dist = sqrt($dist);
         push @d, $dist;
      }
      my $min;
      my $minidx;
      foreach my $i (0 .. $#{$args{centroids}}) {
         if (!defined($min) or $d[$i] < $min) {
            $minidx = $i;
            $min = $d[$i];
         }
      }
      if ($d->{$args{cluster_key}} != $minidx) {
         $changes++;
         $d->{$args{cluster_key}} = $minidx;
      }
  }
  return $changes;
}
         
sub update_centroids {
   my %args = @_;
   my @new_centroids;
   my @cluster_points;
   foreach my $d (@{$args{data}}) {
      foreach my $i (0 .. $#{$d->{$args{coords_key}}}) {
         $new_centroids[ $d->{$args{cluster_key}} ]->[$i] += $d->{$args{coords_key}}[$i];
      }
      $cluster_points[$d->{$args{cluster_key}}]++;
   }
   foreach my $i (0 .. $#{cluster_points}) {
      foreach my $j (0 .. $#{$new_centroids[$i]}) {
         $new_centroids[$i]->[$j] /= $cluster_points[$i];
      }
   }
   return \@new_centroids;
}

use Inline CPP => Config =>
        BUILD_NOISY => 0,
        force_build => 0,
        clean_after_build => 0,
        warnings => 0,
        INC => "-I" . abs_path("./inc") . " -I" . abs_path("./amd_kernel"),
        LIBS => "-L" . abs_path("./amd_kernel") . " -lKernels"
;

use Inline CPP => $code;

sub clusterise_pp { # pure perl version
   my $self = shift;
   my %args = @_;
   my $data = [];
   my $centroids = [];
   foreach my $x (0 .. $args{clusters} - 1) {
      push @$centroids, $args{data}->[$x]{$args{coords_key}};
   }
   my ($changes) = update_clusters(data => $args{data}, centroids => $centroids,
                                    coords_key => $args{coords_key},
                                    cluster_key => $args{cluster_key});
   # not counting the first set as an iteration
   my $iteration = 0;
   while ($iteration++ < $args{maxiter} and $changes > 0) {
      $centroids = update_centroids( data => $args{data}, centroids => $centroids,
                        coords_key => $args{coords_key},
                        cluster_key => $args{cluster_key} );
      $changes = update_clusters( data => $args{data}, centroids => $centroids,
                                    coords_key => $args{coords_key},
                                    cluster_key => $args{cluster_key});
   }
   if ($changes == 0) {
      say "converged in $iteration iterations";
   } else {
      say "unconverged, changes made in last iteration = $changes";
   }
}

sub distance {
   my $self = shift;
   my $centroids = shift;
   my $points = shift;
   my @distances;
   my @sum_distances;
   foreach my $x (0 .. $#{$centroids}) {
      $sum_distances[$x] = 0;
   }
   foreach my $point (@$points) {
      my @d;
      foreach my $c (0 .. $#$centroids) {
         my $d = 0;
         foreach my $x (0 .. $#$point) { # each dimension of the point
            $d += ($point->[$x] - $centroids->[$c][$x])**2;
         }
         $sum_distances[$c] += $d;
         push @d, $d;
      }
      push @distances, \@d;
   }
   return \@distances, \@sum_distances;
}

sub init_centroids {
# implements the K++ centroid initialisation algorithm 
   my $self = shift;
   my $data = shift;
   my $clusters = shift;
   my $c1 = int(rand(scalar(@$data)));
   my @centroids = ( $data->[$c1] );
   foreach my $k (2 .. $clusters) {
      my ($distances, $sum_distances) = $self->distance(\@centroids, $data);
      my $min_idx = 0;
      my $min_sum = $sum_distances->[0];
      foreach my $i (1 .. $#$sum_distances) {
         if ($sum_distances->[$i] < $min_sum) {
            $min_idx = $i;
            $min_sum = $sum_distances->[$i];
         }
      }
      foreach my $d (0 .. $#{$distances}) {
         $distances->[$d][$min_idx] /= $min_sum;
         if ($d > 0) {
            $distances->[$d][$min_idx] += $distances->[$d - 1][$min_idx];
         }
      }
      my $rnd = rand();
      my $new_centroid;
      foreach my $i (0 .. $#{$distances}) {
         if ($distances->[$i][$min_idx] >= $rnd) {
            $new_centroid = $i;
            last;
         }
      }
      push @centroids, $data->[$new_centroid];
   }
   return \@centroids;
}

sub clusterise {
   my $self = shift;
   my %args = @_;
   my $data = [];
   my $changes = 0;
   my $iteration = 0;
   $args{maxiter} = 100 unless defined($args{maxiter}) and $args{maxiter} =~ /^\d+$/;
   my $centroids = $self->init_centroids( $args{ data } , $args{ clusters });
#say "centoids = " . Dumper($centroids);
   if (defined($args{coords_key})) {
      foreach my $d (@{$args{data}}) {
         push @$data, $d->{$args{coords_key}};
      }
   } else {
      $data = $args{data};
   }

   get_me_in_the_mood($centroids, $data) && die;
   $changes = are_we_there_yet();
   while ($iteration++ < $args{maxiter} and $changes > 0) {
      bring_me_closer();
      $changes = are_we_there_yet();
   }
#   if ($changes == 0) {
#      say "converged in $iteration iterations";
#   } else {
#      say "unconverged, changes made in last iteration = $changes";
#   }
   my $clusters = [];
   take_me_home($clusters);

   if (defined($args{cluster_key})) {
      foreach (zip $args{data}, $clusters) {
         my ($d, $c) = @$_;
         $d->{$args{cluster_key}} = $c;
      }
   } else {
      return $clusters;
   }

}

sub centroids {
   my $self = shift;
   my $centroids = [];
   get_centroids($centroids);
   return $centroids;
}

sub DESTROY {
   clean_me_up_im_dirty();
}
1;
