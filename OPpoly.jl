module OPpoly
#
# Module OPpoly contains functions manipulating multivariate polynomials.
# A polynomial is represented as a vector of coefficients.  The indexing
# is arranged (taking a D=3,ord=4 example) as
# c000, c100, c010, c001, c200, c110, c101, c020, c011, c002, c300, c210,
# c201, c120, c111, c102, c030, c021, c012, c003, c400, c310, c301, c220,
# c211, c202, c130, c121, c112, c103, c040, c031, c022, c013, c004
#

export tensorToList, listToTensor, polyrec, polyeval, rateval,
       solveLSPolyCoeffs

# =======================================================================
function tensorToList(t)
#
# Convert between tensor (00,10,01,20,11,02,30,21,12,03,...) and list 
# (1,2,3,...) indexing.
#
# Version:        Changes:
# --------        -------------
# 13.04.2021      Original code.
#
# Version:        Verification:
# --------        -------------
# 13.04.2021      Tested on sample problems up to D=4 and ord=3.
#
# Inputs:
# -------
# t               : D-by-N matrix of integers, with D the dimension.
#
# Outputs:
# --------
# L               : Length N vector of integers, containing the indices.

   D = size(t,1)
   N = size(t,2)

   L = ones(Int,N)
   for ip in 1:N
      ord = sum(t[1:D,ip])
      for id in 1:D
         De = D - id + 1
         L[ip] = L[ip] + binomial((ord-1)+De,De)
         ord = ord - t[id,ip]
      end
   end

   return L

end # tensorToList

# =======================================================================
function listToTensor(L, D)
#
# Convert between list (1,2,3,...) and tensor indexing
# (00,10,01,20,11,02,30,21,12,03,...).
#
# Version:        Changes:
# --------        -------------
# 13.04.2021      Original code.
#
# Version:        Verification:
# --------        -------------
# 13.04.2021      Verified that this inverts tensorToList.
#
# Inputs:
# -------
# L               : Index list, vector of length N.
# D               : Dimension, integer.
#
# Outputs:
# --------
# t               : D-by-N matrix of tensor indices.

   N = length(L)

   # The matrix A is consistent for any dimension D.
   A = zeros(Int,D,D)
   for id = 1:D
      A[id,id:D] = ones(Int,1,D-id+1)
   end

   b = zeros(Int,D,N)
   for ip in 1:N

      Le = L[ip]

      for id in 1:D

         De = D - id + 1

         ind = 1
         sav = 0
         flg = true
         while flg
            if (ind >= Le)
               flg = false
               Le = Le - sav
            else
               b[id,ip] = b[id,ip] + 1
               sav = ind
               ind = binomial(De+b[id,ip],De)
            end
         end

      end

   end

   t = zeros(Int,D,N)
   t[:,:] = A\b

   return t

end # listToTensor

# =======================================================================
function polyrec(x, ord; ptype=0)
#
# Recursively evaluate the normalized terms of a polynomial; that is,
# the value of each term when the coefficient is equal to 1.
#
# Version:        Changes:
# --------        -------------
# 22.04.2021      Original code.
#
# Version:        Verification:
# --------        -------------
# 22.04.2021      Hand-checked the first few terms of a D=2 case.
#
# Inputs:
# -------
# x               : Column vector of length D.
# ord             : The order of individual x components, up to which
#                   the polynomial is to be computed.
# ptype           : Specifies the type of polynomial.
#                   0: Nominal   x[n+1] = x x[n]
#                   1: Chebyshev T[n+1] = 2xT[n] - T[n-1]
#                   2: Legendre  P[n+1] = ((2n+1)xP[n] - nP[n-1])/(n+1)
#                   3: Hermite   H[n+1] = (x H[n] - sq(n)H[n-1])/sq(n+1)
#
# Outputs:
# --------
# psi             : A D-by-ord+1 matrix containing the terms' values
#                   at the given x vector.

   D = size(x,1)
   psi = zeros(D,ord+1)
   psi[:,1] = ones(D)

   if (ord > 0)
      if (ptype == 1)
         # Chebyshev
         for id in 1:D
            psi[id,2] = x[id]
            for n in 1:ord-1
               psi[id,n+2] = 2*x[id]*psi[id,n+1] - psi[id,n]
            end
         end
      elseif (ptype == 2)
         # Legendre
         for id in 1:D
            psi[id,2] = x[id]
            for n in 1:ord-1
               psi[id,n+2] = ((2*n+1)*x[id]*psi[id,n+1] - n*psi[id,n])/(n+1)
            end
         end
      elseif (ptype == 3)
         # Hermite
         for id in 1:D
            psi[id,2] = x[id]
            sqn = 1.0
            for n in 1:ord-1
               sqn1 = sqrt(n+1)
               psi[id,n+2] = (x[id]*psi[id,n+1] - sqn*psi[id,n])/sqn1
               sqn = sqn1  # Avoid recomputing sqrt's.
            end
         end
      else
         # Default to nominal.
         for id in 1:D
            psi[id,2] = x[id]
            for n = 1:ord-1
               psi[id,n+2] = x[id]*psi[id,n+1]
            end
         end
      end
   end # if ord > 0.

   return psi

end # polyrec

# =======================================================================
function polyeval(p, t, x; ptype=0)
#
# Evaluate a polynomial defined by coefficients p of the tensor product
# t at the coordinates x.  This is currently designed to be easy to use,
# not optimal in terms of speed and storage.
#
# Version:        Changes:
# --------        -------------
# 22.04.2021      Original code.
#
# Version:        Verification:
# --------        -------------
# 22.04.2021      
#
# Inputs:
# -------
# p               : Vector of polynomial coefficients.  See the module
#                   header for the convention used for indexing.
# t               : Tensor specifying the x component orders in each
#                   polynomial term.  D-by-N matrix.
# x               : Column vector or matrix; each column contains an x
#                   vector of length D.
# ptype           : Specifies the type of polynomial.  See polyrec.
#
# Outputs:
# --------
# y               : A vector containing the outputs, one scalar output
#                   for each x vector.

   D = size(x,1)   # Dimension of x.
   M = size(x,2)   # Number of points.
   N = size(p,1)   # Number of coefficients.

   ord = maximum(sum(t,dims=1))

   y = zeros(M)

   for ip in 1:M

      # Compute the individual polynomial terms in a recursive way.
      if (D == 1)
         # This is required since the ":" notation is not permitted
         # if x is a scalar.
         psi = polyrec(x[ip],ord,ptype=ptype)
      else
         psi = polyrec(x[:,ip],ord,ptype=ptype)
      end

      for ic in 1:N
         val = 1.0
         for id = 1:D
            val = val*psi[id,t[id,ic]+1]
         end
         y[ip] = y[ip] + p[ic]*val
      end

   end

   return y

end # polyeval: version including explicit input of the sequence.

function polyeval(p, x; ptype=0)
#
# Evaluate a polynomial defined by coefficients p at the coordinates x.
# This is currently designed to be easy to use, not optimal in terms of
# speed and storage.
#
# This version generates the tensor t automatically, based on the
# assumed sequence of polynomials indicated in the module header.
#
# Version:        Changes:
# --------        -------------
# 22.04.2021      Original code.
#
# Version:        Verification:
# --------        -------------
# 22.04.2021      
#

   D = size(x,1)   # Dimension of x.
   N = size(p,1)   # Number of coefficients.

   L = range(1,stop=N)
   t = listToTensor(L,D)

   y = polyeval(p,t,x,ptype=ptype)

   return y

end # polyeval: version with an assumed polynomial sequence.

# =======================================================================
function rateval(p, tp, q, tq, x; ptype=0)
#
# Evaluate a rational function defined by numerator coefficients p and
# denominator coefficients q at the coordinates x.  The order associated
# with each coefficient is given in the matrices tp,tq.
# 
#
# Version:        Changes:
# --------        -------------
# 22.04.2021      Original code.
#
# Version:        Verification:
# --------        -------------
# 22.04.2021      Verified some example calculations.
#
# Inputs:
# -------
# p,q             : Coefficients of numerator and denominator polynomials.
# tp,tq           : Tensors specifying the x component orders in each
#                   polynomial term.  D-by-Np,Nq matrices.
# x               : Column vector or matrix; each column contains an x
#                   vector.
#
# Outputs:
# --------
# y               : A vector containing the outputs, one scalar output
#                   for each x vector.

   yp = polyeval(p,tp,x,ptype=ptype)
   yq = polyeval(q,tq,x,ptype=ptype)

   return (yp ./ yq)

end # rateval: explicit sequence.

function rateval(p, q, x; ptype=0)
#
# Evaluate a rational function defined by numerator coefficients p and
# denominator coefficients q at the coordinates x.
#
# Version:        Changes:
# --------        -------------
# 22.04.2021      Original code.
#
# Version:        Verification:
# --------        -------------
# 22.04.2021      
#

   yp = polyeval(p,x,ptype=ptype)
   yq = polyeval(q,x,ptype=ptype)

   return (yp ./ yq)

end # rateval: implicit sequence.

#=

# =======================================================================
function polyderiv(p, t, k, x; cof=0)
#
# Evaluate the derivatives of a polynomial at specified points.  At the
# moment this supports only nominal polynomials (ptype = 0).  That is,
# if a column of t contains, say, (1, 0, 3, 2) then the polynomial term
# is (x1 x3^3 x4^2).
#
# Version:        Changes:
# --------        -------------
# 18.05.2021      Original code.
#
# Version:        Verification:
# --------        -------------
# 18.05.2021      
#
# Inputs:
# -------
# p               : Vector of polynomial coefficients, length N.  See
#                   the module header for the convention used for
#                   indexing.
# t               : Tensor specifying the x component orders in each
#                   polynomial term.  D-by-N matrix.
# k               : An integer specifying which derivative to take:
#                   1 = x1, 2 = x2, ...
# x               : Column vector or matrix of dimension D-by-M; each
#                   column contains an x vector of length D.
#
# Outputs:
# --------
# pd, td          : Coefficients and tensor describing the derivative
#                   polynomial.

   ptype = 0       # Limited functionality in order to get something
                   # working fast.

   D = size(x,1)   # Dimension of x.
   M = size(x,2)   # Number of points.
   N = size(p,1)   # Number of coefficients.

   ord = maximum(sum(t,dims=1))

   dy = zeros(M,L)

   for ip in 1:M

      # Compute the individual polynomial terms in a recursive way.
      if (D == 1)
         # This is required since the ":" notation is not permitted
         # if x is a scalar.
         psi = polyrec(x[ip],ord,ptype=ptype)
      else
         psi = polyrec(x[:,ip],ord,ptype=ptype)
      end

      for ic in 1:N
         val = 1.0
         for id in 1:D
            val = val*psi[id,t[id,ic]+1]
         end
         y[ip] = y[ip] + p[ic]*val
      end

   end

   return y

end # polyderiv

=#






# =======================================================================
#function polyint()
#
# Evaluate 
#
# Version:        Changes:
# --------        -------------
# 13.04.2021      Original code.
#
# Version:        Verification:
# --------        -------------
# 13.04.2021      
#
# Inputs:
# -------
# 
#
# Outputs:
# --------
# 



#end # polyint

# ===================================================================
function solveLSPolyCoeffs(x, y, tp, tq; ptype=0, wgt=1.0)
#
# Given a list of samples, solves a (weighted) least-squares
# polynomial or rational-function fit.
#
# [Note, this should be augmented to accommodate complex coefficients.]
#
# Version:        Changes:
# --------        -------------
# 22.04.2021      Original code.
#
# Version:        Verification:
# --------        -------------
# 22.04.2021      Verified the fit to a known quadratic rational
#                 polynomial function in two dimensions.
#
# Inputs:
# -------
# x               : D-by-Ns vector of coordinates.
# y               : Length Ns vector of sample values.
# tp,tq           : Tensors specifying the x component orders in each
#                   polynomial term.  D-by-Np,Nq matrices.  Note that
#                   tq should always include one all-zeros term as its
#                   first entry.
# ptype           : Specifies the type of polynomial.  See polyrec.
# wgt             : A vector of weights, the square roots of the weight
#                   matrix diagonal.
#
# Outputs:
# --------
# p,q             : Vector of best-fit polynomial coefficients; p
#                   for the numerator polynomials, q for the
#                   denominator polynomials.

   D  = size(x,1)
   Ns = size(x,2)
   Np = size(tp,2)
   Nq = size(tq,2)
   Nc = Np + Nq

   # The maximum polynomial order in the individual variables.
   op = maximum(tp)
   oq = maximum(tq)
   omax = max(op,oq)

   if (omax == 0)
      # We are dealing with fitting a least-squares constant, not a
      # polynomial.      
      A = ones(Ns,1) .* wgt
      AT = transpose(A)
      p = (AT*A)\(AT*(wgt .* y))
      q = 1.0
   else

      # The matrix in the equation A [p;q] = y.  The "-1" is for the
      # forced constant = 1 term in the denominator.
      A = ones(Ns,Nc-1)  
      for isamp in 1:Ns

         # Compute the individual polynomial terms in a recursive way.
         psi = polyrec(x[:,isamp],omax,ptype=ptype)

         for ip in 1:Np
            for id in 1:D
               A[isamp,ip] = A[isamp,ip]*psi[id,tp[id,ip]+1]
            end
         end
         for iq in 1:Nq-1  # Nq-1: Neglect the constant 1.0 in the denominator.
            ind = Np + iq
            for id in 1:D
               A[isamp,ind] = A[isamp,ind]*psi[id,tq[id,iq+1]+1]
            end
            A[isamp,ind] = -A[isamp,ind]*y[isamp]
         end
      end

      A = A .* wgt

      AT = A'
      c = (AT*A)\(AT*(wgt .* y))
      p = c[1:Np]
      if (oq > 0)
         q = [1.0;c[Np+1:Nc-1]]
      else
         q = 1.0
      end

   end

   return p, q

end  # solveLSPolyCoeffs

end # module
