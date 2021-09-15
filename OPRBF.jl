module OPRBF
#
# Module OPRBF implements functions related to radial basis function fitting
# and interpolation.
#

export RBFmat, RBFfit, RBFkernel, RBFeval, RBFcondition, RBFderiv

using LinearAlgebra

# ===================================================================
function RBFmat(x, s; ker=1)
#
# Compute the matrix used in evaluating the RBF fit.
#
# Version:        Changes:
# --------        -------------
# 13.06.2021      Original code.
#
# Version:        Verification:
# --------        -------------
# 13.06.2021      
#
# Inputs:
# -------
# x               : Reference points, Nd-by-Np matrix.
# s               : Radial basis function scale parameter.
# ker             : Flag selecting the type of kernel function: see
#                   RBFkernel.
#
# Outputs:
# --------
# M               : The matrix, Np-by-Np.

   Np = size(x,2)

   # Set up the system of equations.
   M = zeros(Np,Np)
   for irow = 1:Np
      M[irow,irow] = RBFkernel(0.0, 1.0, s, ker=ker)
      for icol = irow+1:Np
         r = norm(x[:,irow] - x[:,icol])
         M[irow,icol] = RBFkernel(r, 1.0, s, ker=ker)
         M[icol,irow] = M[irow,icol]
      end
   end

   return M

end  # RBFmat
function RBFmat(x1, x2, s; ker=1)
#
# This version of RBFmat generates the matrix between two sets of
# points that may be different.
#
# Version:        Changes:
# --------        -------------
# 24.06.2021      Original code.
#
# Version:        Verification:
# --------        -------------
# 24.06.2021      
#
# Inputs:
# -------
# x1,x2           : Reference points, Nd-by-N1 and Nd-by-N2 matrices.
# s               : Radial basis function scale parameter.
# ker             : Flag selecting the type of kernel function: see
#                   RBFkernel.
#
# Outputs:
# --------
# M               : The matrix, N1-by-N2.

   N1 = size(x1,2)
   N2 = size(x2,2)

   # Set up the system of equations.
   M = zeros(N1,N2)
   for irow = 1:N1
      for icol = 1:N2
         r = norm(x1[:,irow] - x2[:,icol])
         M[irow,icol] = RBFkernel(r, 1.0, s, ker=ker)
      end
   end

   return M

end  # RBFmat

# ===================================================================
function RBFfit(x, y, s; ker=1, con=false)
#
# Fit a radial basis function of the specified kernel to the given points.
#
# [Note, could also make a version that takes different x and xp points,
#  where Nx > Nxp, performing a least-squares solution at the x points
#  for RBF coefficients at the xp points.]
#
# Version:        Changes:
# --------        -------------
# 10.05.2021      Original code.
#
# Version:        Verification:
# --------        -------------
# 10.05.2021      Checked an example 2D surface-fitting problem.
#
# Inputs:
# -------
# x               : Reference points, Nd-by-Np matrix.
# y               : Scalar values at the reference points, Vector of
#                   length Np.
# s               : Radial basis function scale parameter.
# ker             : Flag selecting the type of kernel function: see
#                   RBFkernel.
# con             : Set to true to compute and return the matrix 
#                   condition number.
#
# Outputs:
# --------
# w               : Weights, vector of length Np.

   Np = size(x,2)

   # Set up the system of equations.
   M = RBFmat(x, s, ker=ker)

   if (con)
      c = cond(M)
   else
      c = 0.0
   end

   w = M \ y

   return w, c

end  # RBFfit

# ===================================================================
function RBFkernel(r, w, s; ker=1)
#
# Compute the kernel at the given points.
#
# Version:        Changes:
# --------        -------------
# 10.05.2021      Original code.
#
# Version:        Verification:
# --------        -------------
# 10.05.2021      Checked an example 2D surface-fitting problem.
#
# Inputs:
# -------
# r               : Distances at which to evaluate the kernel function.
#                   Vector of length Np.
# w               : Kernel function weights.
# s               : Kernel function scale parameter.
# ker             : Flag specifying the type of kernel:
#                   1: Gaussian.
#                   2: Multiquadratic.
#                   3: Inverse multiquadratic.
#                   4: Student T with nu = 1.
#
# Outputs:
# --------
# y               : Kernel function values.  Vector of length Np.

   if (ker == 1)      # Gaussian
      y = w .* exp.(-(s .* r) .^ 2)
   elseif (ker == 2)  # Quadratic
      y = w .* sqrt.(1.0 .+ (s .* r) .^ 2)
   elseif (ker == 3)  # Inverse quadratic
      y = w ./ sqrt.(1.0 .+ (s .* r) .^ 2)
   elseif (ker == 4)  # Student T, nu = 1
      y = s .* w ./ (pi*(1.0 .+ (s .* r) .^ 2))
   end

   return y

end  # RBFkernel

# ===================================================================
function RBFeval(x, xp, w, s; ker=1)
#
# Compute the value of the function at the given points.
#
# Version:        Changes:
# --------        -------------
# 10.05.2021      Original code.
#
# Version:        Verification:
# --------        -------------
# 10.05.2021      Checked an example 2D surface-fitting problem.
#
# Inputs:
# -------
# x               : Points at which the function is desired.  Nd-by-
#                   Nx matrix.
# xp              : Points at which the weights are defined.  Nd-by-
#                   Np matrix.
# w               : Kernel function weights.  Vector of length Np.
# s               : Kernel function scale parameter.
# ker             : Flag specifying the type of kernel: see RBFkernel.
#
# Outputs:
# --------
# f               : Function values.  Vector of length Nx.

   Nx = size(x,2)
   Np = size(xp,2)
   Nd = size(x,1)

   f = zeros(Nx)
   for ix in 1:Nx
      r = zeros(Np)
      for id in 1:Nd
         r = r .+ (x[id,ix] .- xp[id,:]).^2
      end
      r = sqrt.(r)
      y = RBFkernel(r, w, s; ker=ker)
      f[ix] = sum(y)

   end

   return f

end  # RBFeval

# ===================================================================
function RBFcondition(x, y, sg; clow=5.0e4, chigh=2.0e5, ker=1)
#
# Iterate on the scale parameter s until the matrix condition number
# lies within the specified bounds.
#
# Version:        Changes:
# --------        -------------
# 12.05.2021      Original code.
#
# Version:        Verification:
# --------        -------------
# 12.05.2021      Checked an example 2D surface-fitting problem.
#
# Inputs:
# -------
# x               : Referance points, Nd-by-Np matrix.
# y               : Scalar values at the reference points, Vector of
#                   length Np.
# sg              : Guess for the radial basis function scale parameter.
# clow            : Lower bound for the matrix condition number.
# chigh           : Upper bound for the matrix condition number.
# ker             : Flag selecting the type of kernel function: see
#                   RBFkernel.
#
# Outputs:
# --------
# w               : Weights, vector of length Np.
# s               : Value of the scale parameter satisfying the 
#                   condition number.
# c               : The final condition number.

   s = deepcopy(sg)
   w,c = RBFfit(x, y, s; ker=ker, con=true)

   # First bracket the desired condition number.
   if (c < clow)

      # The condition number is low.  Make the scale parameter smaller,
      # which makes the basis function broader.  (Sorry for the confusion,
      # but this is the typical definition.)
      sl = deepcopy(s)
      sh = deepcopy(s)
      iter = 0
      while (c < clow) && (iter < 10)
         sl = 0.5*sl
         w,c = RBFfit(x, y, sl; ker=ker, con=true)
         iter = iter + 1'
      end

      if (c > chigh)
         flg = false
      else
         # The upper bound now lies within the acceptable range.  If
         # iter reached its limit, well, this is anyways the best that
         # we are going to get, so no additional logic is needed.
         s = deepcopy(sl)
         flg = true
      end

   elseif (c > chigh)

      # The condition number is high.  Make the scale parameter larger,
      # which makes the basis function narrower.
      sl = deepcopy(s)
      sh = deepcopy(s)
      iter = 0
      while (c > chigh) && (iter < 10)
         sh = 2.0*sh
         w,c = RBFfit(x, y, sh; ker=ker, con=true)
         iter = iter + 1
      end

      if (c < clow)
         flg = false
      else
         # The lower bound now lies within the acceptable range.
         s = deepcopy(sh)
         flg = true
      end

   else
      # The initial s lies within the acceptable range.
      flg = true
   end

   # Now, if flg is false, cl and ch bracket the acceptable range.  Do
   # a bisecting search.
   iter = 0
   while (!flg) && (iter < 10)
      s = 0.5*(sh + sl)
      w,c = RBFfit(x, y, s; ker=ker, con=true)
      if (c < clow)
         # The condition number is low.  Make the scale parameter smaller.
         sh = deepcopy(s)
      elseif (c > chigh)
         # The condition number is high.  Make the scale parameter larger.
         sl = deepcopy(s)
      else
         flg = true
      end
      iter = iter + 1
   end

   return w,s,c

end  # RBFcondition

# ===================================================================
function RBFderiv(x, xp, w, s, k; ker=1)
#
# Compute the derivatives of the radial basis functions with respect
# to each of the coordinates, in a specified sequence.
#
# Note that this function does not compute df/dx_1, df/dx_2, ... but
# rather df/dx_1, d^2f/dx_1dx_2, ...  If the former is desired,
# then call this function multiple times with the appropriate values
# of k.
#
# Version:        Changes:
# --------        -------------
# 16.05.2021      Original code.
#
# Version:        Verification:
# --------        -------------
# 16.05.2021      Checked a 2D example problem, comparing with finite
#                 difference using RBFeval.
#
# Inputs:
# -------
# x               : Points at which the function is desired.  Nd-by-
#                   Nx matrix.
# xp              : Points at which the weights are defined.  Nd-by-
#                   Np matrix.
# w               : Kernel function weights.
# s               : Kernel function scale parameter.
# k               : Sequence in which derivatives are to be taken.
#                   Vector of length Ndim, containing the integers
#                   1:Nd in the desired order of differentiation.
#                   Example: d^3f/dx_2 dx_1 dx_3; k = [2, 1, 3]
#                   (It doesn't HAVE to contain all the integers from
#                   1 to Ndim, but there cannot be any repeated
#                   indices.)
# ker             : Flag specifying the type of kernel:
#                   1: Gaussian.
#                   2: Multiquadratic.
#                   3: Inverse multiquadratic.
#                   4: Student T with nu = 1.
#
# Outputs:
# --------
# f               : Values of the RBFs, same as a call to RBFeval. 
#                   These are computed here regardless, so might as
#                   well return them as output.
# df              : Matrix of derivatives, dimensions Nk-by-Nx, 
#                   containing df/dx_k[1], d^2f/dx_k[1] dx_k[2], ...
#                   d^Nk f/dx_k[1] dx_k[2] ... dx_k[Nk].

   Nx = size(x,2)
   Np = size(xp,2)
   Nd = size(x,1)
   Nk = size(k,1)

   # Build up the formulas recursively.
   f  = zeros(Nx)
   df = zeros(Nk,Nx)

   for ix in 1:Nx
      r = zeros(Np)
      for id in 1:Nd
         r = r .+ (x[id,ix] .- xp[id,:]).^2
      end
      r = sqrt.(r)
      y = RBFkernel(r, w, s; ker=ker)
      if (ker == 1)      # Gaussian
         v  =  s^2
         c  = -2.0
         dc =  0.0
      elseif (ker == 2)  # Quadratic
         v  =  (s^2) ./ (1.0 .+ (s .* r).^2)
         c  =  1.0
         dc = -2.0
      elseif (ker == 3)  # Inverse quadratic
         v  =  (s^2) ./ (1.0 .+ (s .* r).^2)
         c  = -1.0
         dc = -2.0
      elseif (ker == 4)  # Student T.
         v  =  (s^2) ./ (1.0 .+ (s .* r).^2)
         c  = -2.0
         dc = -2.0
      end
      f[ix] = sum(y)
      for ik in 1:Nk
         y = y .* (c * v .* (x[k[ik],ix] .- xp[k[ik],:]))
         df[ik,ix] = sum(y)
         c = c + dc
      end
      
   end

   return f, df

end # RBFderiv

end  # module.