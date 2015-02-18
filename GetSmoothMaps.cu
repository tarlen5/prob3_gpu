/*
  GetSmoothMaps.cu

  author: Timothy C. Arlen
          tca3@psu.edu

  date:   7 Feb 2015

  Implements the procedure of gettting smooth oscillation probability
  maps for a given set of oscillation parameters and energy/coszen
  bins through a specific earth density model.

*/

#include "EarthDensity.h"
#include "BinUtils.h"
#include "mosc.h"
#include "mosc3.h"
#include "utils.h"

#include <stdio.h>
#include <time.h>
#include <math.h>
#include <fstream>
#include <iomanip>

// For now, follow probGpu.cu, but maybe change this later...
static double h_dm[3][3];
static double h_mix[3][3][2];

const double kmTOcm = 1.0e5;

#define VERBOSE false

__host__ double DefinePath(EarthDensity* earth, double cz, double prod_height,
                           double rDetector)
/*
  prod_height - Atmospheric production height [cm]
  rDetector - Detector radius [cm]
 */
{
  double path_length = 0.0;
  double depth = earth->get_DetectorDepth()*kmTOcm;

  //double ProductionHeight = prod_height*kmTOcm;

  if(cz < 0) {
    path_length = sqrt((rDetector + prod_height +depth)*(rDetector + prod_height +depth)
                       - (rDetector*rDetector)*( 1 - cz*cz)) - rDetector*cz;
  } else {
    double kappa = (depth + prod_height)/rDetector;
    path_length = rDetector*sqrt(cz*cz - 1 + (1 + kappa)*(1 + kappa)) - rDetector*cz;
  }

  return path_length;

}


//GetEarthLayerInfo(earth_model_file,czbins_fine,ncz_fine);
__host__ int GetEarthLayerInfo(char* earth_model_file, double* czcen_fine, int ncz_fine,
                               double*& densityInLayer,double*& distanceInLayer,
                               int*& numberOfLayers)
{

  // These should go into the Earth model file
  double detector_depth = 2.0; double prod_height = 20.0*kmTOcm;

  // Initialize Earth Model (host) and then fill density, distance
  // within each layer for each path, defined by coszenith direction
  EarthDensity* earth_model = new EarthDensity(earth_model_file, detector_depth);
  double rDetector = earth_model->get_RDetector()*kmTOcm;
  int maxLayers = earth_model->get_MaxLayers();

  densityInLayer = (double*)malloc(ncz_fine*maxLayers*sizeof(double));
  distanceInLayer = (double*)malloc(ncz_fine*maxLayers*sizeof(double));
  numberOfLayers = (int*)malloc(ncz_fine*sizeof(int));

  for (int i=0; i<ncz_fine; i++) {
    double coszen = czcen_fine[i];
    double pathLength = DefinePath(earth_model, coszen, prod_height, rDetector);
    earth_model->SetDensityProfile( coszen, pathLength, prod_height );

    *(numberOfLayers+i) = earth_model->get_LayersTraversed();
    //printf("coszen: %f, layers traversed: %d, path length: %f\n",coszen,
    //       *(numberOfLayers+i), pathLength/kmTOcm);
    for (int j=0; j < *(numberOfLayers+i); j++) {
      double density = earth_model->get_DensityInLayer(j)*
        earth_model->get_ElectronFractionInLayer(j);
      *(densityInLayer + i*maxLayers + j) = density;

      double distance = earth_model->get_DistanceAcrossLayer(j)/kmTOcm;
      *(distanceInLayer + i*maxLayers + j) = distance;
      //printf("  >> Layer: %d, density: %f, distance: %f\n",j,*(densityInLayer + i*maxLayers + j),
      //     *(distanceInLayer + i*maxLayers + j));
    }
  }

  delete earth_model;

  return maxLayers;

}


__host__ void DefineEarthLayerArrays(char* earth_model_file, double* czcen_fine, int nczbins_fine,
                                     double*& d_densityInLayer,double*& d_distanceInLayer,
                                     int*& d_numberOfLayers,int& maxLayers)
{

  // 2D arrays, but use 1D and keep track using careful indexing
  // These are defined in GetEarthLayerInfo()
  double* densityInLayer = NULL;
  double* distanceInLayer = NULL;
  int* numberOfLayers = NULL;
  maxLayers = GetEarthLayerInfo(earth_model_file,czcen_fine,nczbins_fine,
                                densityInLayer,distanceInLayer,numberOfLayers);
  printf("  -->all path lengths defined.\n");

  // Initialize memory on gpu for density/distance in each Earth Layer, then max layers
  size_t dens_size = nczbins_fine*maxLayers*sizeof(double);
  checkCudaErrors(cudaMalloc((void**)&d_densityInLayer,dens_size));
  checkCudaErrors(cudaMemcpy(d_densityInLayer,densityInLayer,dens_size,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc((void**)&d_distanceInLayer,dens_size));
  checkCudaErrors(cudaMemcpy(d_distanceInLayer,distanceInLayer,dens_size,cudaMemcpyHostToDevice));

  size_t layer_size = nczbins_fine*sizeof(int);
  checkCudaErrors(cudaMalloc((void**)&d_numberOfLayers,layer_size));
  checkCudaErrors(cudaMemcpy(d_numberOfLayers,numberOfLayers,layer_size,cudaMemcpyHostToDevice));

  // Free Host memory:
  free(densityInLayer);
  free(distanceInLayer);
  free(numberOfLayers);

}


__host__ void InitializeFineAndSmoothMaps(double**& h_smooth_maps, double**& d_smooth_maps,
                                          double**& h_fine_maps, double**& d_fine_maps,
                                          const int& nbins, const int& nbins_fine_tot,
                                          const int& nMaps)
{


  // Initialize maps on device global memory:
  // First, it is necessary to get the pointer of device pointers on host memory,
  // then copy it to device memory:
  //int nbins = nebins*nczbins;
  //double** h_smooth_maps = (double**)malloc(nMaps*sizeof(double*));
  h_smooth_maps = (double**)malloc(nMaps*sizeof(double*));
  printf("host smooth maps array allocated...\n");
  for(int i=0; i<nMaps; i++) {
    double* d_map;
    checkCudaErrors(cudaMalloc((void**) &d_map,nbins*sizeof(double)));
    checkCudaErrors(cudaMemset(d_map,0.0,nbins*sizeof(double)));
    *(h_smooth_maps + i) = d_map;
  }
  checkCudaErrors(cudaMalloc((void***) &d_smooth_maps, nMaps*sizeof(double*)));
  checkCudaErrors(cudaMemcpy(d_smooth_maps,h_smooth_maps,nMaps*sizeof(double*),
                             cudaMemcpyHostToDevice));


  // initialize fine maps on host/device
  //double** h_fine_maps = (double**)malloc(nMaps*sizeof(double*));
  h_fine_maps = (double**)malloc(nMaps*sizeof(double*));
  for(int i=0; i<nMaps; i++) {
    double* d_fine_map;
    checkCudaErrors(cudaMalloc((void**)&d_fine_map,nbins_fine_tot*sizeof(double)));
    *(h_fine_maps + i) = d_fine_map;
  }
  //double** d_fine_maps;
  checkCudaErrors(cudaMalloc((void***)&d_fine_maps,nMaps*sizeof(double*)));
  checkCudaErrors(cudaMemcpy(d_fine_maps,h_fine_maps,nMaps*sizeof(double*),
                             cudaMemcpyHostToDevice));


  // All maps initialized now...

}


__host__ void init_mixing_matrix(double dm21,double dm32, double sin12,double sin13,
                                 double sin23, double deltacp,bool verbose=false)
/*
  NOTE: As in all parts of this code, expects dm21 and dm32 in eV^2
  and sinij to be units of sin(theta_{ij})
*/
{

  // these functions are all described in mosc.h/mosc.cu
  setMatterFlavor(nue_type);
  setmix_sin(sin12,sin23,sin13,deltacp,h_mix);
  setmass(dm21,dm32,h_dm);

  if (verbose) {
    printf("dm21,dm32   : %f %f \n",dm21,dm32);
    printf("s12,s23,s31 : %f %f %f \n",sin12,sin23,sin13);
    printf("h_dm  : %f %f %f \n",h_dm[0][0],h_dm[0][1],h_dm[0][2]);
    printf("h_dm  : %f %f %f \n",h_dm[1][0],h_dm[1][1],h_dm[1][2]);
    printf("h_dm  : %f %f %f \n",h_dm[2][0],h_dm[2][1],h_dm[2][2]);
    //***********
    //**********
    printf("h_mix : %f %f %f \n",h_mix[0][0][0],h_mix[0][1][0],h_mix[0][2][0]);
    printf("h_mix : %f %f %f \n",h_mix[1][0][0],h_mix[1][1][0],h_mix[1][2][0]);
    printf("h_mix : %f %f %f \n",h_mix[2][0][0],h_mix[2][1][0],h_mix[2][2][0]);
    printf("h_mix : %f %f %f \n",h_mix[0][0][1],h_mix[0][1][1],h_mix[0][2][1]);
    printf("h_mix : %f %f %f \n",h_mix[1][0][1],h_mix[1][1][1],h_mix[1][2][1]);
    printf("h_mix : %f %f %f \n",h_mix[2][0][1],h_mix[2][1][1],h_mix[2][2][1]);
    //***********
  }
}

__host__ void setMNS(double dm_solar,double dm_atm,double x12,double x13,double x23,
                     double deltacp,bool verbose)
/*
  \params:
    - xij = sin^2(\theta_{ij})
    - dm_solar,dm_atm are in units of eV^2
 */

{

  // NOTE: does not support values of x_ij given in sin^2(2*theta_ij)
  double sin12 = sqrt(x12);
  double sin13 = sqrt(x13);
  double sin23 = sqrt(x23);

  init_mixing_matrix(dm_solar,dm_atm,sin12,sin13,sin23,deltacp,verbose);

}


// the main propagate kernel
__global__ void propagate(double** d_fine_maps,double** d_smooth_maps,
//__global__ void propagate(double** d_smooth_maps,
                          double d_dm[3][3], double d_mix[3][3][2],
                          const double* const d_ecen_fine, const double* const d_czcen_fine,
                          const int nebins_fine, const int nczbins_fine,
                          const int nebins, const int nczbins, const int maxLayers,
                          const int* const d_numberOfLayers, const double* const d_densityInLayer,
                          const double* const d_distanceInLayer)
{

  const int2 thread_2D_pos = make_int2(blockIdx.x*blockDim.x + threadIdx.x,
                                       blockIdx.y*blockDim.y + threadIdx.y);

  // ensure we don't access memory outside of bounds!
  if(thread_2D_pos.x >= nczbins_fine || thread_2D_pos.y >= nebins_fine) return;
  const int thread_1D_pos = thread_2D_pos.y*nczbins_fine + thread_2D_pos.x;

  int eidx = thread_2D_pos.y;
  int czidx = thread_2D_pos.x;

  //int eidx = threadIdx.x; //threadIdx.x; // energy index
  //int czidx = blockIdx.x; // coszen index
  int kNuBar;
  if(blockIdx.z == 0) kNuBar = 1;
  else if(blockIdx.z == 1) kNuBar=-1;
  else {
    printf("ERROR! Invalid value of blockIdx.z: %d. Must be 0 or 1 only\n",blockIdx.z);
  }

  bool kUseMassEstates = false;

  double TransitionMatrix[3][3][2];
  double TransitionProduct[3][3][2];
  double TransitionTemp[3][3][2];
  double RawInputPsi[3][2];
  double OutputPsi[3][2];
  double Probability[3][3];

  clear_complex_matrix( TransitionMatrix );
  clear_complex_matrix( TransitionProduct );
  clear_complex_matrix( TransitionTemp );
  clear_probabilities( Probability );

  int layers = *(d_numberOfLayers + czidx);
  double energy = d_ecen_fine[eidx];
  //double coszen = d_czcen_fine[czidx];
  for( int i=0; i<layers; i++) {
    double density = *(d_densityInLayer + czidx*maxLayers + i);
    double distance = *(d_distanceInLayer + czidx*maxLayers + i);

    get_transition_matrix( kNuBar,
                           energy,
                           density,
                           distance,
                           TransitionMatrix,
                           0.0,
                           d_mix,
                           d_dm);

    if(i==0) { copy_complex_matrix(TransitionMatrix, TransitionProduct);
    } else {
      clear_complex_matrix( TransitionTemp );
      multiply_complex_matrix( TransitionMatrix, TransitionProduct, TransitionTemp );
      copy_complex_matrix( TransitionTemp, TransitionProduct );
    }
  } // end layer loop


  // loop on neutrino types, and compute probability for neutrino i:
  // We actually don't care about nutau -> anything since the flux there is zero!
  for( unsigned i=0; i<2; i++) {
    for ( unsigned j = 0; j < 3; j++ ) {
      RawInputPsi[j][0] = 0.0;
      RawInputPsi[j][1] = 0.0;
    }

    if( kUseMassEstates ) convert_from_mass_eigenstate( i+1, kNuBar,  RawInputPsi, d_mix );
    else RawInputPsi[i][0] = 1.0;

    multiply_complex_matvec( TransitionProduct, RawInputPsi, OutputPsi );
    Probability[i][0] += OutputPsi[0][0] * OutputPsi[0][0] + OutputPsi[0][1]*OutputPsi[0][1];
    Probability[i][1] += OutputPsi[1][0] * OutputPsi[1][0] + OutputPsi[1][1]*OutputPsi[1][1];
    Probability[i][2] += OutputPsi[2][0] * OutputPsi[2][0] + OutputPsi[2][1]*OutputPsi[2][1];

  }//end of neutrino loop


  ////////////////////////////////////////////////////
  ///// PUT IN ATOMIC ADD HERE FOR d_smooth_maps /////
  ////////////////////////////////////////////////////
  int efctr = nebins_fine/nebins;
  int czfctr = nczbins_fine/nczbins;
  int eidx_smooth = eidx/efctr;
  int czidx_smooth = czidx/czfctr;
  double scale = double(efctr*czfctr);
  for (int i=0;i<2;i++) {
    int iMap = 0;
    if (kNuBar == 1) iMap = i*3;
    else iMap = 6 + i*3;

    for (unsigned to_nu=0; to_nu<3; to_nu++) {
      double prob = Probability[i][to_nu];
      //*(*(d_fine_maps+iMap+to_nu) + thread_1D_pos) = prob;
      atomicAdd((*(d_smooth_maps + iMap + to_nu) + eidx_smooth*nczbins + czidx_smooth),prob/scale);
    }
  }

}


void GetSmoothMaps(double** smooth_maps, const int nMaps,
                   double* const ebins, double* const czbins,
                   const int nebins, const int nczbins, const int efctr, const int czfctr,
                   const double theta12, const double theta13, const double theta23,
                   const double dm_solar, const double dm_atm,
                   const double deltacp, char* earth_model_file)//, unsigned MAX_THREADS=1024)
/*
  Calculates the average oscillation probability in the bin of the map defined by
  the bin edges (ebins,czbins). The averaging is done by oversampling the oscillation
  weight in each bin defined by efactr, czfctr.

  \params:
    * smooth_maps - 2D array of smoothed oscillation map, to be calculated by this function.
      Rows are of ebins, and cols are of czbins.
    * nMaps - number of smooth_maps addresses within smooth_maps variable
    * ebins - energy bin edges [GeV] of smoothed map
    * czbins - coszen bin edges of smoothed map
    * nebins, nczbins - number of energy and coszen bins in (ebins,czbins) respectively.
    * efctr - factor to oversample the energy bins by to get the smoothed map
    * czfctr - factor to oversample the coszen bins by to get the smoothed map.
    * thetaij - sin(theta_{ij})^2 value
    * dm_solar, dm_atm - solar and atmospheric mass splitting in [eV^2]
    * deltacp - deltacp value in [rad]
    * earth_model_file - file to initialize EarthDensity class.
 */
{

  int nczbins_fine = czfctr*nczbins;
  int nebins_fine = efctr*nebins;
  double* ebins_fine = OverSampleBinning(ebins,nebins,efctr);
  double* czbins_fine = OverSampleBinning(czbins,nczbins,czfctr);
  double* ecen_fine = get_bin_centers(ebins_fine,nebins_fine);
  double* czcen_fine = get_bin_centers(czbins_fine,nczbins_fine);
  double* ecen = get_bin_centers(ebins,nebins);
  double* czcen = get_bin_centers(czbins,nczbins);

  // These 4 variables will be initialized in DefineEarthLayerArrays()
  double *d_densityInLayer, *d_distanceInLayer;
  int* d_numberOfLayers;
  int maxLayers = 0;
  DefineEarthLayerArrays(earth_model_file, czcen_fine, nczbins_fine, d_densityInLayer,
                         d_distanceInLayer,d_numberOfLayers,maxLayers);


  // Initlialize all maps here:
  int nbins = nebins*nczbins;
  int nbins_fine_tot = nebins_fine*nczbins_fine;
  double **h_smooth_maps, **d_smooth_maps, **h_fine_maps, **d_fine_maps;
  InitializeFineAndSmoothMaps(h_smooth_maps, d_smooth_maps, h_fine_maps, d_fine_maps,
                              nbins, nbins_fine_tot, nMaps);


  // set h_dm and h_mix and then copy them to device:
  setMNS(dm_solar,dm_atm,theta12,theta13,theta23,deltacp,VERBOSE);

  typedef double dmArray[3];
  dmArray* d_dm;
  size_t dmsize = 3*3*sizeof(double);
  checkCudaErrors(cudaMalloc((void**) &d_dm, dmsize));
  checkCudaErrors(cudaMemcpy(d_dm,h_dm,dmsize,cudaMemcpyHostToDevice));

  typedef double mixArray[3][2];
  mixArray* d_mix;
  size_t mixsize = 3*3*2*sizeof(double);
  checkCudaErrors(cudaMalloc((void**) &d_mix, mixsize));
  checkCudaErrors(cudaMemcpy(d_mix,h_mix,mixsize,cudaMemcpyHostToDevice));

  // Finally setup ecen_fine, czcen_fine on device
  double* d_ecen_fine; double* d_czcen_fine;
  size_t esize = nebins_fine*sizeof(double);
  size_t czsize = nczbins_fine*sizeof(double);
  checkCudaErrors(cudaMalloc((void**)&d_ecen_fine,esize));
  checkCudaErrors(cudaMemcpy(d_ecen_fine,ecen_fine,esize,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc((void**)&d_czcen_fine,czsize));
  checkCudaErrors(cudaMemcpy(d_czcen_fine,czcen_fine,czsize,cudaMemcpyHostToDevice));


  // Now set up kernel for launch: can we optimize this with shared
  // memory?
  //cudaDeviceProp cuda_prop;
  //checkCudaErrors(cudaGetDeviceProperties(&cuda_prop,0)); int
  //maxThreads = cuda_prop.maxThreadsPerBlock; printf(" --max threads
  //per block: %d \n",maxThreads);

  // Launch 20x20 thread blocks, and grid_size.z=2 to handle nu/nubar
  const dim3 block_size(20,20,1);
  const dim3 grid_size(nczbins_fine/block_size.x + 1, nebins_fine/block_size.y + 1,2);
  printf("Launching kernel...\n");
  propagate<<<grid_size,block_size>>>(d_fine_maps,d_smooth_maps,d_dm,d_mix,
  //propagate<<<grid_size,block_size>>>(d_smooth_maps,d_dm,d_mix,
                                      d_ecen_fine,d_czcen_fine,
                                      nebins_fine,nczbins_fine,nebins,nczbins,maxLayers,
                                      d_numberOfLayers,d_densityInLayer,d_distanceInLayer);
  checkCudaErrors( cudaPeekAtLastError() );
  checkCudaErrors( cudaDeviceSynchronize() );

  printf("cudaMemcpy from Device to Host... \n");

  // FINE MAPS TEST: copy and then free all memory for fine maps:
  checkCudaErrors(cudaMemcpy(h_fine_maps,d_fine_maps,nMaps*sizeof(double*),
                             cudaMemcpyDeviceToHost));

  for(int i=0; i<nMaps; i++) {
    double* h_map = (double*)malloc(nbins_fine_tot*sizeof(double));
    double* d_map = *(h_fine_maps + i);
    checkCudaErrors(cudaMemcpy(h_map,d_map,nbins_fine_tot*sizeof(double),
                               cudaMemcpyDeviceToHost));
    *(h_fine_maps + i) = h_map;
    checkCudaErrors(cudaFree(d_map));
  }
  checkCudaErrors(cudaFree(d_fine_maps));


  checkCudaErrors(cudaMemcpy(h_smooth_maps,d_smooth_maps,nMaps*sizeof(double*),
                             cudaMemcpyDeviceToHost));
  // Smooth maps: copy and free memory...
  for(int i=0; i<nMaps; i++) {
    double* h_map = (double*)malloc(nbins*sizeof(double));
    double* d_map = *(h_smooth_maps + i);
    checkCudaErrors(cudaMemcpy(h_map,d_map,nbins*sizeof(double),
                               cudaMemcpyDeviceToHost));
    *(h_smooth_maps + i) = h_map;
    checkCudaErrors(cudaFree(d_map));
  }
  checkCudaErrors(cudaFree(d_smooth_maps));


  checkCudaErrors(cudaFree(d_densityInLayer));
  checkCudaErrors(cudaFree(d_distanceInLayer));
  checkCudaErrors(cudaFree(d_numberOfLayers));

  checkCudaErrors(cudaFree(d_dm));
  checkCudaErrors(cudaFree(d_mix));

  checkCudaErrors(cudaFree(d_ecen_fine));
  checkCudaErrors(cudaFree(d_czcen_fine));

  // REMOVE IN PROD VERSION: ////
  bool plotBool = false;
  if (plotBool) {
    //Write bins to file:
    /*ofstream fh1("ebins.out");
    for(int i=0; i<=nebins_fine; i++) {
      fh1<<ebins_fine[i]<<" ";
    }
    fh1<<"\n";
    fh1.close();
    fh1.open("czbins.out");
    for(int i=0; i<=nczbins_fine; i++) {
      fh1<<czbins_fine[i]<<" ";
    }
    fh1<<"\n";
    fh1.close(); */

    ofstream fh_bins("ebins_sm.out");
    for(int i=0; i<=nebins; i++) {
      fh_bins<<ebins[i]<<" ";
    }
    fh_bins<<"\n";
    fh_bins.close();
    fh_bins.open("czbins_sm.out");
    for(int i=0; i<=nczbins; i++) {
      fh_bins<<czbins[i]<<" ";
    }
    fh_bins<<"\n";
    fh_bins.close();


    // Write values to file:
    for(int iMap=0; iMap<nMaps; iMap++) {
      ofstream fh;
      char s_map[20];
      sprintf(s_map,"%d",iMap);
      //string filename = "fineMap_"+std::string(s_map)+".out";
      string filename = "smoothMap_"+std::string(s_map)+".out";
      printf(">>Saving to file: %s\n",filename.c_str());
      fh.open(filename.c_str());
      //for (int i=0; i<nebins_fine; i++) {
      for (int i=0; i<nebins; i++) {
        //for (int j=0; j<nczbins_fine; j++) {
        for (int j=0; j<nczbins; j++) {
          fh << std::setprecision(15)<< ecen[i] <<" "<< czcen[j]<<" "
             <<*(*(h_smooth_maps+iMap)+i*nczbins+j)<<"\n";
        }
      }
      fh.close();
      }

  }
  ///////////////////////////////////////////////////////////


  ///////////////////////// WARNING////////////////////////////
  // NOTE: Don't forget to free individual h_fine_maps and h_smooth_maps!
  /////////////////////////////////////////////////////////////

  // Free all remaining host memory:
  free(h_fine_maps);
  free(h_smooth_maps);
  free(ebins_fine);
  free(czbins_fine);
  free(ecen_fine);
  free(czcen_fine);
  free(ecen);
  free(czcen);

}



int main(int argc, char** argv)
{

  double emax = 80.0; double emin=1.0;
  double czmax = 0.0; double czmin = -1.0;
  int negy = 40; int ncz = 21;
  int nebins = negy-1; int nczbins = ncz-1;
  double* ebins = get_logarithmic_spaced(emin,emax,negy);
  double* czbins = get_linear_spaced(czmin,czmax,ncz);

  int efctr = 13; int czfctr = 12;
  // Akhmedov settings:
  //double theta12 = 0.312; double theta13 = 0.025; double theta23 = 0.42;
  //double dm2  = 7.6E-5;  double DM2 = 2.35E-3; double dcp = 0.0;

  // Pisa Oscillation Defaults:
  double sinSq12 = 0.307043940651;
  double sinSq13 = 0.02420065643;
  double sinSq23 = 0.390009;
  double dm2 = 7.54e-5;
  double DM2 = (0.00246 - dm2);
  double dcp = 0.0;

  char* earth_file = "PREM_4layer.dat";

  clock_t start, stop;
  start = clock();

  printf("starting clock...\n");

  ////////////////////////////////////////////////////////////////////
  // What I will do right now is this:
  //   smooth_maps will be a pointer to 12 pointers each of which
  //   hold the data for a nu_i -> nu_f (smoothed) probability map.
  //   Following order will be maintained:
  //     nu_e -> (0) nu_e, (1) nu_mu, (2) nu_tau
  //     nu_mu -> (3) nu_e, (4) nu_mu, (5) nu_tau
  //     nu_ebar -> (6) nu_ebar, (7) nu_mubar, (8) nu_taubar
  //     nu_mubar -> (9) nu_ebar, (10) nu_mubar, (11) nu_taubar
  //   where (i) is the index of the smooth_maps pointer of pointers.
  ////////////////////////////////////////////////////////////////////
  int nMaps = 12;
  double** smooth_maps = (double**)malloc(nMaps*sizeof(double*));
  for (int i=0; i<nMaps; i++) {
    double* prob_map = (double*)malloc(nebins*nczbins*sizeof(double));
    *(smooth_maps + i) = prob_map;
  }

  printf("Getting smoothed maps...\n");
  GetSmoothMaps(smooth_maps,nMaps,ebins,czbins,nebins,nczbins,efctr,czfctr,sinSq12,sinSq13,
                sinSq23,dm2,DM2,dcp,earth_file);

  printf("Freeing all host smoothed maps: \n");
  for(int i=0; i<nMaps; i++) free(*(smooth_maps + i));
  free(smooth_maps);

  free(czbins);
  free(ebins);

  stop =  clock();
  printf("\n  Time elapsed (GetSmoothMap): %f sec\n",((double)(stop - start))/CLOCKS_PER_SEC);

  return 0;

}
