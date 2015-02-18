#include "mosc.h"
#include <stdio.h>

#define elec (0)
#define muon (1)
#define tau  (2)
#define re (0)
#define im (1)

//#define ZERO_CP
static int matrixtype = standard_type;

/* Flag to tell us if we're doing nu_e or nu_sterile matter effects */
static NuType matterFlavor = nue_type;
static double putMix[3][3][2];


__host__ void setMatterFlavor(int flavor)
{
  if (flavor == nue_type) matterFlavor = nue_type;
  else if (flavor == sterile_type) matterFlavor = sterile_type;
  else {
    //fprintf(stderr, "setMatterFlavor: flavor=%d", flavor);
    //moscerr("setMatterFlavor: Illegal flavor.");
  }
}

__host__ void setmix_sin(double s12,double s23,double s13,double dcp,
                         double Mix[3][3][2])
{
  double c12,c23,c13,sd,cd;

  if ( s12>1.0 ) s12=1.0;
  if ( s23>1.0 ) s23=1.0;
  if ( s13>1.0 ) s13=1.0;
  if ( cd >1.0 ) cd =1.0;

  sd = sin( dcp );
  cd = cos( dcp );

  c12 = sqrt(1.0-s12*s12);
  c23 = sqrt(1.0-s23*s23);
  c13 = sqrt(1.0-s13*s13);

  if ( matrixtype == standard_type )
    {
      Mix[0][0][re] =  c12*c13;
      Mix[0][0][im] =  0.0;
      Mix[0][1][re] =  s12*c13;
      Mix[0][1][im] =  0.0;
      Mix[0][2][re] =  s13*cd;
      Mix[0][2][im] = -s13*sd;
      Mix[1][0][re] = -s12*c23-c12*s23*s13*cd;
      Mix[1][0][im] =         -c12*s23*s13*sd;
      Mix[1][1][re] =  c12*c23-s12*s23*s13*cd;
      Mix[1][1][im] =         -s12*s23*s13*sd;
      Mix[1][2][re] =  s23*c13;
      Mix[1][2][im] =  0.0;
      Mix[2][0][re] =  s12*s23-c12*c23*s13*cd;
      Mix[2][0][im] =         -c12*c23*s13*sd;
      Mix[2][1][re] = -c12*s23-s12*c23*s13*cd;
      Mix[2][1][im] =         -s12*c23*s13*sd;
      Mix[2][2][re] =  c23*c13;
      Mix[2][2][im] =  0.0;
    }
  else
    {
      Mix[0][0][re] =  c12;
      Mix[0][0][im] =  0.0;
      Mix[0][1][re] =  s12*c23;
      Mix[0][1][im] =  0.0;
      Mix[0][2][re] =  s12*s23;
      Mix[0][2][im] =  0.0;
      Mix[1][0][re] = -s12*c13;
      Mix[1][0][im] =  0.0;
      Mix[1][1][re] =  c12*c13*c23+s13*s23*cd;
      Mix[1][1][im] =              s13*s23*sd;
      Mix[1][2][re] =  c12*c13*s23-s13*c23*cd;
      Mix[1][2][im] =             -s13*c23*sd;
      Mix[2][0][re] = -s12*s13;
      Mix[2][0][im] =  0.0;
      Mix[2][1][re] =  c12*s13*c23-c13*s23*cd;
      Mix[2][1][im] =             -c13*s23*sd;
      Mix[2][2][re] =  c12*s13*s23+c13*c23*cd;
      Mix[2][2][im] =              c13*c23*sd;
    }
}

__host__ void setmass(double dms21, double dms23, double dmVacVac[][3])
{
  double delta=5.0e-9;
  double mVac[3];

  mVac[0] = 0.0;
  mVac[1] = dms21;
  mVac[2] = dms21+dms23;

  /* Break any degeneracies */
  if (dms21==0.0) mVac[0] -= delta;
  if (dms23==0.0) mVac[2] += delta;

  dmVacVac[0][0] = dmVacVac[1][1] = dmVacVac[2][2] = 0.0;
  dmVacVac[0][1] = mVac[0]-mVac[1]; dmVacVac[1][0] = -dmVacVac[0][1];
  dmVacVac[0][2] = mVac[0]-mVac[2]; dmVacVac[2][0] = -dmVacVac[0][2];
  dmVacVac[1][2] = mVac[1]-mVac[2]; dmVacVac[2][1] = -dmVacVac[1][2];
}


/***********************************************************************
  getM
  Compute the matter-mass vector M, dM = M_i-M_j and
  and dMimj. type<0 means anti-neutrinos type>0 means "real" neutrinos
***********************************************************************/
__device__ void getM(double Enu, double rho,
                     double Mix[][3][2], double dmVacVac[][3], int antitype,
                     double dmMatMat[][3], double dmMatVac[][3])
{
  int i, j, k;
  double alpha, beta, gamma, fac=0.0, arg, tmp;
  double alphaV, betaV, gammaV, argV, tmpV;
  double theta0, theta1, theta2;
  double theta0V, theta1V, theta2V;
  double mMatU[3], mMatV[3], mMat[3];
  double tworttwoGf = 1.52588e-4;

  /* Equations (22) fro Barger et.al.*/
  /* Reverse the sign of the potential depending on neutrino type */
  //if (matterFlavor == nue_type) {
  /* If we're doing matter effects for electron neutrinos */
  if (antitype<0) fac =  tworttwoGf*Enu*rho; /* Anti-neutrinos */
  else        fac = -tworttwoGf*Enu*rho; /* Real-neutrinos */
  //}
  //else if (matterFlavor == sterile_type) {
  /* If we're doing matter effects for sterile neutrinos */
  //if (antitype<0) fac = -0.5*tworttwoGf*Enu*rho; /* Anti-neutrinos */

  //   else        fac =  0.5*tworttwoGf*Enu*rho; /* Real-neutrinos */
  // }
  /* The strategy to sort out the three roots is to compute the vacuum
   * mass the same way as the "matter" masses are computed then to sort
   * the results according to the input vacuum masses
   */

  alpha  = fac + dmVacVac[0][1] + dmVacVac[0][2];
  alphaV = dmVacVac[0][1] + dmVacVac[0][2];

#ifndef ZERO_CP
  beta = dmVacVac[0][1]*dmVacVac[0][2] +
    fac*(dmVacVac[0][1]*(1.0 - Mix[elec][1][re]*Mix[elec][1][re] -
                         Mix[elec][1][im]*Mix[elec][1][im]) +
         dmVacVac[0][2]*(1.0 - Mix[elec][2][re]*Mix[elec][2][re] -
                         Mix[elec][2][im]*Mix[elec][2][im]));
  betaV = dmVacVac[0][1]*dmVacVac[0][2];

#else
  beta = dmVacVac[0][1]*dmVacVac[0][2] +
    fac*(dmVacVac[0][1]*(1.0 - Mix[elec][1][re]*Mix[elec][1][re]) +
         dmVacVac[0][2]*(1.0- Mix[elec][2][re]*Mix[elec][2][re]));
  betaV = dmVacVac[0][1]*dmVacVac[0][2];
#endif

#ifndef ZERO_CP
  gamma = fac*dmVacVac[0][1]*dmVacVac[0][2]*
    (Mix[elec][0][re]*Mix[elec][0][re]+Mix[elec][0][im]*Mix[elec][0][im]);
  gammaV = 0.0;
#else
  gamma = fac*dmVacVac[0][1]*dmVacVac[0][2]*
    (Mix[elec][0][re]*Mix[elec][0][re]);
  gammaV = 0.0;
#endif

  /* Compute the argument of the arc-cosine */
  tmp = alpha*alpha-3.0*beta;
  tmpV = alphaV*alphaV-3.0*betaV;
  if (tmp<0.0) {
    // fprintf(stderr, "getM: alpha^2-3*beta < 0 !\n");
    tmp = 0.0;
  }

  /* Equation (21) */
  arg = (2.0*alpha*alpha*alpha-9.0*alpha*beta+27.0*gamma)/
    (2.0*sqrt(tmp*tmp*tmp));
  if (fabs(arg)>1.0) arg = arg/fabs(arg);
  argV = (2.0*alphaV*alphaV*alphaV-9.0*alphaV*betaV+27.0*gammaV)/
    (2.0*sqrt(tmpV*tmpV*tmpV));
  if (fabs(argV)>1.0) argV = argV/fabs(argV);

  /* These are the three roots the paper refers to */
  theta0 = acos(arg)/3.0;
  theta1 = theta0-(2.0*M_PI/3.0);
  theta2 = theta0+(2.0*M_PI/3.0);
  theta0V = acos(argV)/3.0;
  theta1V = theta0V-(2.0*M_PI/3.0);
  theta2V = theta0V+(2.0*M_PI/3.0);

  mMatU[0] = mMatU[1] = mMatU[2] = -(2.0/3.0)*sqrt(tmp);
  mMatU[0] *= cos(theta0); mMatU[1] *= cos(theta1); mMatU[2] *= cos(theta2);

  tmp = dmVacVac[0][0] - alpha/3.0;
  mMatU[0] += tmp; mMatU[1] += tmp; mMatU[2] += tmp;
  mMatV[0] = mMatV[1] = mMatV[2] = -(2.0/3.0)*sqrt(tmpV);
  mMatV[0] *= cos(theta0V); mMatV[1] *= cos(theta1V); mMatV[2] *= cos(theta2V);
  tmpV = dmVacVac[0][0] - alphaV/3.0;

  mMatV[0] += tmpV; mMatV[1] += tmpV; mMatV[2] += tmpV;

  /* Sort according to which reproduce the vaccum eigenstates */
  for (i=0; i<3; i++) {
    tmpV = fabs(dmVacVac[i][0]-mMatV[0]);
    k = 0;
    for (j=1; j<3; j++) {
      tmp = fabs(dmVacVac[i][0]-mMatV[j]);
      if (tmp<tmpV) {
        k = j;
        tmpV = tmp;
      }
    }
    mMat[i] = mMatU[k];
  }

  for (i=0; i<3; i++) {
    for (j=0; j<3; j++) {
      dmMatMat[i][j] = mMat[i] - mMat[j];
      dmMatVac[i][j] = mMat[i] - dmVacVac[j][0];
    }
 }
}

/***********************************************************************
 getA
 Calculate the transition amplitude matrix A (equation 10)
***********************************************************************/
__device__ void getA(double L, double E, double rho,
                     double Mix[][3][2], double dmMatVac[][3],
                     double dmMatMat[][3], int antitype, double A[3][3][2],
                     double phase_offset)
{

  /*
    DARN - looks like this is all junk...more debugging needed...
  */

  //int n, m, i, j, k;
  double /*fac=0.0,*/ arg, c, s;
  // TCA ADDITION: set equal to 0!
  double X[3][3][2] = {0.0};
  double product[3][3][3][2] = {0.0};
  /* (1/2)*(1/(h_bar*c)) in units of GeV/(eV^2-km) */
  const double LoEfac = 2.534;

  if ( phase_offset==0.0 )
    {
      get_product(L, E, rho, Mix, dmMatVac, dmMatMat, antitype, product);
    }

  /////////////// product is JUNK /////////////

  for (int i=0; i<3; i++){
    for (int j=0; j<3; j++) {
  //printf(" product[%d][%d]: %f, %f\n",i,j,*product[i][j][0],*product[i][j][1]);
  //printf(" A[%d][%d]: %f, %f\n",i,j,A[i][j][0],A[i][j][1]);
    }
  }

  /* Make the sum with the exponential factor */
  //cudaMemset(X, 0, 3*3*2*sizeof(double));
  //memset(X, 0, 3*3*2*sizeof(double));
  for (int k=0; k<3; k++)
    {
      arg = -LoEfac*dmMatVac[k][0]*L/E;
      if ( k==2 ) arg += phase_offset ;
      c = cos(arg);
      s = sin(arg);
      for (int i=0; i<3; i++)
        {
          for (int j=0; j<3; j++)
            {
#ifndef ZERO_CP
              X[i][j][re] += c*product[i][j][k][re] - s*product[i][j][k][im];
              X[i][j][im] += c*product[i][j][k][im] + s*product[i][j][k][re];
#else
              X[i][j][re] += c*product[i][j][k][re];
              X[i][j][im] += s*product[i][j][k][re];
#endif
            }
        }
    }


  /* Compute the product with the mixing matrices */
  for(int i=0; i < 3; i++)
    for(int j = 0; j < 3; j++)
      for(int k = 0; k < 2; k++)
        A[i][j][k] = 0;

  for (int n=0; n<3; n++) {
    for (int m=0; m<3; m++) {
      for (int i=0; i<3; i++) {
        for (int j=0; j<3; j++) {
#ifndef ZERO_CP
          A[n][m][re] +=
            Mix[n][i][re]*X[i][j][re]*Mix[m][j][re] +
            Mix[n][i][re]*X[i][j][im]*Mix[m][j][im] +
            Mix[n][i][im]*X[i][j][re]*Mix[m][j][im] -
            Mix[n][i][im]*X[i][j][im]*Mix[m][j][re];
          //printf("\nregret %f %f %f",Mix[n][i][re], X[i][j][im], Mix[m][j][im]);
          A[n][m][im] +=
            Mix[n][i][im]*X[i][j][im]*Mix[m][j][im] +
            Mix[n][i][im]*X[i][j][re]*Mix[m][j][re] +
            Mix[n][i][re]*X[i][j][im]*Mix[m][j][re] -
            Mix[n][i][re]*X[i][j][re]*Mix[m][j][im];
#else
          A[n][m][re] +=
            Mix[n][i][re]*X[i][j][re]*Mix[m][j][re];
          A[n][m][im] +=
            Mix[n][i][re]*X[i][j][im]*Mix[m][j][re];
#endif
          //printf("\n %i %i %i A %f", n, m, re, A[n][m][re]);
        }
      }
    }
  }

  //printf("(getA) Aout: %f\n",A[0][0][0]);

}


__device__ void get_product(double L, double E, double rho,double Mix[][3][2],
                            double dmMatVac[][3], double dmMatMat[][3],
                            int antitype,
                            double product[][3][3][2])
{

  double fac=0.0;
  double twoEHmM[3][3][3][2];
  double tworttwoGf = 1.52588e-4;

  /* (1/2)*(1/(h_bar*c)) in units of GeV/(eV^2-km) */
  /* Reverse the sign of the potential depending on neutrino type */
  //if (matterFlavor == nue_type) {

  /* If we're doing matter effects for electron neutrinos */
  if (antitype<0) fac =  tworttwoGf*E*rho; /* Anti-neutrinos */
  else        fac = -tworttwoGf*E*rho; /* Real-neutrinos */
  //  }

  /*
      else if (matterFlavor == sterile_type) {
      // If we're doing matter effects for sterile neutrinos
      if (antitype<0) fac = -0.5*tworttwoGf*E*rho; // Anti-neutrinos
      else        fac =  0.5*tworttwoGf*E*rho; // Real-neutrinos
      } */

  /* Calculate the matrix 2EH-M_j */
  for (int n=0; n<3; n++) {
    for (int m=0; m<3; m++) {

#ifndef ZERO_CP
      twoEHmM[n][m][0][re] =
        -fac*(Mix[0][n][re]*Mix[0][m][re]+Mix[0][n][im]*Mix[0][m][im]);
      twoEHmM[n][m][0][im] =
        -fac*(Mix[0][n][re]*Mix[0][m][im]-Mix[0][n][im]*Mix[0][m][re]);

      twoEHmM[n][m][1][re] = twoEHmM[n][m][2][re] = twoEHmM[n][m][0][re];
      twoEHmM[n][m][1][im] = twoEHmM[n][m][2][im] = twoEHmM[n][m][0][im];

#else

      twoEHmM[n][m][0][re] =
        -fac*(Mix[0][n][re]*Mix[0][m][re]);
      twoEHmM[n][m][0][im] = 0 ;
      twoEHmM[n][m][1][re] = twoEHmM[n][m][2][re] = twoEHmM[n][m][0][re];
      twoEHmM[n][m][1][im] = twoEHmM[n][m][2][im] = twoEHmM[n][m][0][im];

#endif

      if (n==m) for (int j=0; j<3; j++)
                  twoEHmM[n][m][j][re] -= dmMatVac[j][n];
    }
  }

  /* Calculate the product in eq.(10) of twoEHmM for j!=k */
  for (int i=0; i<3; i++) {
    for (int j=0; j<3; j++) {
      for (int k=0; k<3; k++) {

#ifndef ZERO_CP

        product[i][j][0][re] +=
          twoEHmM[i][k][1][re]*twoEHmM[k][j][2][re] -
          twoEHmM[i][k][1][im]*twoEHmM[k][j][2][im];
        product[i][j][0][im] +=
          twoEHmM[i][k][1][re]*twoEHmM[k][j][2][im] +
          twoEHmM[i][k][1][im]*twoEHmM[k][j][2][re];
        product[i][j][1][re] +=
          twoEHmM[i][k][2][re]*twoEHmM[k][j][0][re] -
          twoEHmM[i][k][2][im]*twoEHmM[k][j][0][im];
        product[i][j][1][im] +=
          twoEHmM[i][k][2][re]*twoEHmM[k][j][0][im] +
          twoEHmM[i][k][2][im]*twoEHmM[k][j][0][re];
        product[i][j][2][re] +=
          twoEHmM[i][k][0][re]*twoEHmM[k][j][1][re] -
          twoEHmM[i][k][0][im]*twoEHmM[k][j][1][im];
        product[i][j][2][im] +=
          twoEHmM[i][k][0][re]*twoEHmM[k][j][1][im] +
          twoEHmM[i][k][0][im]*twoEHmM[k][j][1][re];

#else
        product[i][j][0][re] +=
          twoEHmM[i][k][1][re]*twoEHmM[k][j][2][re];
        product[i][j][1][re] +=
          twoEHmM[i][k][2][re]*twoEHmM[k][j][0][re];
        product[i][j][2][re] +=
          twoEHmM[i][k][0][re]*twoEHmM[k][j][1][re];

#endif
      }
#ifndef ZERO_CP

      product[i][j][0][re] /= (dmMatMat[0][1]*dmMatMat[0][2]);
      product[i][j][0][im] /= (dmMatMat[0][1]*dmMatMat[0][2]);
      product[i][j][1][re] /= (dmMatMat[1][2]*dmMatMat[1][0]);
      product[i][j][1][im] /= (dmMatMat[1][2]*dmMatMat[1][0]);
      product[i][j][2][re] /= (dmMatMat[2][0]*dmMatMat[2][1]);
      product[i][j][2][im] /= (dmMatMat[2][0]*dmMatMat[2][1]);

#else
      product[i][j][0][re] /= (dmMatMat[0][1]*dmMatMat[0][2]);
      product[i][j][1][re] /= (dmMatMat[1][2]*dmMatMat[1][0]);
      product[i][j][2][re] /= (dmMatMat[2][0]*dmMatMat[2][1]);

#endif
    }
  }
}
