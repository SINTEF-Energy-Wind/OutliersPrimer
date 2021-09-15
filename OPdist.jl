module OPdist
#
# Module OPdist contains functions that create and manipulate probability
# distributions.
#

export binit, moments, tdist

using SpecialFunctions
using LinearAlgebra

using ..OPccmt
using ..OPpoly

# ===================================================================
function binit(dat,wgt,Nb,mflag)
#
# Bins it.  Creates a distribution from input data.
#
# Version:        Changes:
# --------        -------------
# 21.05.2020      Original code.
#
# Version:        Verification:
# --------        -------------
# 21.05.2020      
#
# Inputs:
# -------
# dat             : Raw data, input as an N-by-M array, M coordinates of
#                   N dimensions.
# wgt             : Weights to attach to each point; in a typical
#                   application this will be 1, and this can be input as
#                   a single scalar.
# Nb              : Number of bins along each dimension.
# mflag           : Method flag.  1=nearest bin, 2=linear weight.
#
# Outputs:
# --------
# dis             : The binned distribution N-by-prod(Nb).
# xb              : N-by-3 array containing xmin, dx, xmax for each dimension.

   M = size(dat,2)
   N = size(dat,1)

   prodNb = prod(Nb)

   # Get the bins.
   xb = zeros(N,3)
   for idim = 1:N
      mx = maximum(dat[idim,:])
      mn = minimum(dat[idim,:])
      mxmn = mx - mn
      rng = 1.1*mxmn
      dx = rng/(Nb[idim]-1)
      xb[idim,3] = mx + 0.05*mxmn
      xb[idim,1] = mn - 0.05*mxmn
      xb[idim,2] = dx
   end

#print(dat)
#print("\n")
#print(xb)
#print("\n")

   # Bin the data.
   ic, wc = nearestCells(dat,xb)

#print(ic)
#print("\n")
#print(wc)

   dis = zeros(prodNb)
   if (size(wgt,1) == size(dat,2))
      if (mflag == 1)  # Nearest neighbor.
         for ip = 1:M
            jnk, jc = findmax(wc[:,ip])
            dis[ic[jc,ip]] = dis[ic[jc,ip]] + wgt[ip]
         end
      elseif (mflag == 2)  # Linear weight.
         for ip = 1:M
            dis[ic[:,ip]] = dis[ic[:,ip]] + wc[:,ip]*wgt[ip]
         end
      end
   elseif (size(wgt,1) == 1)
      if (mflag == 1)  # Nearest neighbor.
         for ip = 1:M
            jnk, jc = findmax(wc[:,ip])
#print([ip jnk jc ic[jc] dis[ic[jc]]])
#print("\n")
            dis[ic[jc,ip]] = dis[ic[jc,ip]] + wgt
         end
      elseif (mflag == 2)  # Linear weight.
         for ip = 1:M
            dis[ic[:,ip]] = dis[ic[:,ip]] + wc[:,ip]*wgt
         end
      end
   end

   return dis, xb

end  # binit

# ===================================================================
function moments(xc,phi,n)
#
# Compute the nth degree moment of the binned input distribution.
#
# Version:        Changes:
# --------        -------------
# 08.01.2021      Original code.
#
# Version:        Verification:
# --------        -------------
# 08.01.2021      
#
# Inputs:
# -------
# xc              : Ndim-by-Np array of bin coordinates.
# phi             : Probability within each bin.  This does not
#                   need to be normalized, it may be raw counts.
# n               : Order of the moment to be computed.
#
# Outputs:
# --------
# m               : The order-n moments.  Vector of length Ndim^n.

   Ndim = size(xc,1)
   Np = size(xc,2)   

   tot = sum(phi)

   # Calculate the mean.
   mu = zeros(Ndim)
   for ip = 1:Np
      mu = mu + xc[:,ip] .* phi[ip]
   end
   mu = mu / tot

#print(mu,"\n")

   if (n == 0)
      m = 1.0
   elseif (n == 1)
      m = zeros(Ndim)
   else
      # Calculate the desired moments.
      xb = zeros(n,3)
      xb[:,1] .= 1
      xb[:,2] .= 1
      xb[:,3] .= Ndim
      Ncell = Ndim^n
      ic = range(1,length=Ncell)
      ks = xFromCell(ic,xb)
      ks = convert.(Int,round.(ks))

#print(ks,"\n")

      m = zeros(Ncell)
      for ip = 1:Np
         for jc = 1:Ncell
            mul = 1.0
#print("-----------------------\n")
            for jord = 1:n
               mul = mul*(xc[ks[jord,jc],ip] - mu[ks[jord,jc]])
#print(ip," ",jc," ",jord," ",xc[ks[jord,jc],ip]," ",mu[ks[jord,jc]]," ",mul,"\n")
            end
            m[jc] = m[jc] + mul*phi[ip]
         end
      end
      m = m / tot
   end

   return m

end  # moments

# ===================================================================
function tdist(x,mu,Sig,nu)
#
# Compute a t distribution at the specified points.
#
# Version:        Changes:
# --------        -------------
# 18.01.2021      Original code.
#
# Version:        Verification:
# --------        -------------
# 18.01.2021      
#
# Inputs:
# -------
# x               : The points at which to compute the t distribution.
#                   Ndim rows by Np columns.
# mu              : The mean vector.
# Sig             : The matrix Sigma, noting that for nu > 2 the
#                   covariance is (nu/(nu-2)) Sigma.
# nu              : The t distribution DOF parameter.  Approaches
#                   Gaussian as nu -> Inf.  Heavier tails for small nu.
#
# Outputs:
# --------
# t               : The t distribution values.

   d = size(x,1)
   Np = size(x,2)
   detS = det(Sig)
   invS = inv(Sig)
   xc = x .- mu

   gg = loggamma(0.5*(nu + d)) - loggamma(0.5*nu)
   c1 = (((pi*nu)^d)*detS)^(-0.5)
   c2 = -0.5*(nu + d)

   t = zeros(typeof(x[1]),Np)
   for ip in 1:Np
      t[ip] = exp(gg)*c1*
              ((1.0 + (transpose(xc[:,ip])*invS*xc[:,ip])/nu)^c2)
   end

   return t

end  # tdist



end  # module
