/*
  Utilities for binning and checking bin characteristics.

  author: Timothy C. Arlen
          tca3@psu.edu

  date:   7 Feb 2015

  WARNING: With all these functions, the return values are pointers
  malloc'd within the given function. Therefore, each returned pointer
  to memory (or 1D array) must be free'd to avoid memory leaks!

*/

#ifndef __BINUTILS_H__
#define __BINUTILS_H__

#include <math.h>
#include <stdlib.h>

double* get_linear_spaced(double min, double max, unsigned nedges)
{
  double* edges = (double*) malloc(nedges*sizeof(double));
  double delta = (max - min)/(nedges-1);
  for(unsigned i=0; i<nedges; i++) edges[i] = min + double(i)*delta;
  return edges;
}

double* get_logarithmic_spaced(double min, double max, unsigned nedges)
{
  double* edges = (double*) malloc(nedges*sizeof(double));
  double delta = (log10(max) - log10(min))/(nedges-1);
  for(unsigned i=0; i<nedges; i++) edges[i] = min*pow(10.0,delta*double(i));
  return edges;
}

bool is_linear(double* bins,unsigned nbins)
{
  double precision = 1.0e-12*fabs(bins[nbins-1] - bins[0]);
  double linear_bins[nbins];
  double delta = (bins[nbins-1] - bins[0])/double(nbins-1);
  for (unsigned i=0; i<nbins; i++) {
    linear_bins[i] = bins[0] + double(i)*delta;
    if (fabs(bins[i] - linear_bins[i]) > precision) return false;
  }
  return true;
}

bool is_logarithmic(double* bins, unsigned nbins)
{
  double precision = 1.0e-12*fabs(bins[nbins-1] - bins[0]);
  double log_bins[nbins];
  double delta = (log10(bins[nbins-1]) - log10(bins[0]))/(nbins-1);
  for (unsigned i=0; i<nbins; i++) {
    log_bins[i] = bins[0]*pow(10.0,delta*double(i));
    if (fabs(bins[i] - log_bins[i])/log_bins[i] > precision) return false;
  }
  return true;
}

double* get_bin_centers(double* edges, unsigned nbins)
{
  double* cen = (double*)malloc(nbins*sizeof(double));
  //unsigned nbins = nedges -1;
  if(is_linear(edges,nbins)) {
    double delta = (edges[1] - edges[0])/2.0;
    cen  = get_linear_spaced(edges[0]+delta,edges[nbins]-delta,nbins);
  } else if(is_logarithmic(edges,nbins)) {
    double bmin = sqrt((edges[1]*edges[0]));
    double bmax = sqrt((edges[nbins-1]*edges[nbins]));
    cen = get_logarithmic_spaced(bmin,bmax,nbins);
  } else {
    printf("ERROR: bins are neither equally spaced in log or linear space!\n");
    for(int i=0; i<nbins; i++) printf("  edges[%d] = %f\n",i,edges[i]);
    exit(EXIT_FAILURE);
  }
  return cen;
}

double* OverSampleBinning(double* bins, unsigned nbins, unsigned factor)
/*
  bins - (old) bin edges
  nbins - (old) number of bin edges
  factor - oversampling factor
 */
{
  unsigned nbins_fine = nbins*factor;
  double* bins_fine = NULL;

  if (is_linear(bins,nbins)) {
    bins_fine = get_linear_spaced(bins[0],bins[nbins],nbins_fine+1);
  } else if(is_logarithmic(bins,nbins)) {
    bins_fine = get_logarithmic_spaced(bins[0],bins[nbins],nbins_fine+1);
  } else {
    printf("ERROR: bins are neither eqally spaced in log or linear space!\n");
    for(int i=0; i<nbins; i++) printf("  bins[%d] = %f\n",i,bins[i]);
    exit(EXIT_FAILURE);
  }

  return bins_fine;

}


#endif // __BINUTILS_H__
