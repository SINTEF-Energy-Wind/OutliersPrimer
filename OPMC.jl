module OPMC
#
# Monte-Carlo type methods for stochastic dynamical systems.
#

export matchPoints, poissonDisk, selectPoints, resample

using Random

using ..OPccmt

# ===================================================================
function matchPoints(xv, xz)
#
# Representing a collection of particles as a chaos expansion
# requires mapping the coordinates from the physical domain to a
# reference domain.  Given the coordinates of the physical points,
# and a set of reference coordinates, map the physical to the
# reference coordinates.
#
# The algorithm used is "farthest nearest neighbor".  It is a greedy
# sequential algorithm assigning closest point pairs; but it does so
# in an order starting from the point whose minimum distance among
# all candidate matching points is greatest.  This prevents the case
# where all the close points are assigned first, leaving the last
# couple matches between points on opposite sides of the domain.
#
# Here we will call the points in physical space "v" and in reference
# space "z".
#
# Version:        Changes:
# --------        -------------
# 08.07.2021      Original code.
#
# Version:        Verification:
# --------        -------------
# 08.07.2021      Checked a 2D example problem.
#
# Inputs:
# -------
# xv              : A D-by-N matrix of points in physical space,
#                   where D is the number of dimensions and N the
#                   number of points.
# xz              : The matching points in the reference space, also
#                   a D-by-N matrix.
#
# Outputs:
# --------
# k               : A 2-by-N matrix of indices.  The first row is
#                   the index of a v point, and the second of a z
#                   point, such that each column identifies a pair.

   D = size(xv,1)  # Number of dimensions.
   N = size(xv,2)  # Number of points.

   # Compute the coordinate-wise cumulative distributions.  Future 
   # improvement: this can likely be accelerated by the use of a
   # 1D sorting algorithm.
   Pv = zeros(D,N)
   Pz = zeros(D,N)
   for ip in 1:N
      for id in 1:D
         ind = (xv[id,:] .< xv[id,ip])
         Pv[id,ip] = sum(ind)
         ind = (xz[id,:] .< xz[id,ip])   # Future improvement: can compute
         Pz[id,ip] = sum(ind)            # this once upfront.
      end
   end

   # Build the matrix of distances between Pv and Pz points.
   Svz  = zeros(N,N)
   for ir in 1:N
      for ic in 1:N
         for id in 1:D
            Svz[ir,ic] = Svz[ir,ic] + (Pv[id,ir] - Pz[id,ic])^2
         end
         Svz[ir,ic] = sqrt(Svz[ir,ic])
      end
   end

   rng = range(1,stop=N)
   indr = ones(Bool,N)
   indc = ones(Bool,N)
   k = zeros(Int,2,N)
   for ip in 1:N
      nn = N+1-ip
      minr = zeros(nn)
      minc = zeros(nn)
      rngr = rng[indr]
      rngc = rng[indc]
      for jp in 1:nn
         minr[jp] = minimum(Svz[rngr[jp],rngc])
         minc[jp] = minimum(Svz[rngr,rngc[jp]])
      end
      irmax = argmax(minr)
      icmax = argmax(minc)
      rmax = rngr[irmax]
      cmax = rngc[icmax]

      if (minr[irmax] > minc[icmax])
         imin = argmin(Svz[rmax,rngc])
         cmin = rngc[imin]
         k[1,ip] = rmax
         k[2,ip] = cmin
         indr[rmax] = false
         indc[cmin] = false
      else
         imin = argmin(Svz[rngr,cmax])
         rmin = rngr[imin]
         k[1,ip] = rmin
         k[2,ip] = cmax
         indr[rmin] = false
         indc[cmax] = false
      end
   end

   return k

end  # function matchPoints.

# ===================================================================
function poissonDisk(D, N, k; rseed=-1)
#
# Fills space with a set of points that are randomly located, but a
# minimum distance from all other points in the domain.  This is
# known as "Poisson disk sampling", or "blue noise sampling".
#
# The domain is the D-dimensional unit cube.  The coordinates can be
# easily scaled afterwards.
#
# Bridson R.  Fast Poisson disk sampling in arbitrary dimensions.
# Proceedings of ACM SIGGRAPH 2007, p 22.
#
# Roberts M.  http://extremelearning.com.au/an-improved-version-of-
# bridsons-algorithm-n-for-poisson-disc-sampling/
#
# Version:        Changes:
# --------        -------------
# 26.07.2021      Original code.
#
# Version:        Verification:
# --------        -------------
# 26.07.2021      Plotted 2D cases, showing proper execution.  Ran
#                 3D and 4D cases.
#
# Inputs:
# -------
# D               : The dimension of the space.
# N               : The number of cells along each dimension.
# k               : The maximum number of trials before rejection.
#
# Outputs:
# --------
# x               : Coordinates of the points.

   if (rseed > 0)
      Random.seed!(rseed)
   end

   del = sqrt(eps())
   sqD = sqrt(D)

   # Initialize a background grid.
   M = N^D               # Number of cells.
   dx = 1.0/N
   r  = dx*sqrt(D)

   xb = zeros(D,3)
   xb[:,1] .= 0.5*dx
   xb[:,2] .= dx
   xb[:,3] .= 1.0 - 0.5*dx

   ic = range(1,stop=M)
   xc = xFromCell(ic,xb)

   # Dynamically allocate the number of points.
   Narray = 1

   act = zeros(Bool,Narray)     # Active point flag.
   xs  = zeros(D,Narray)        # Point coordinates.
   cx  = zeros(Int,Narray)      # Point cell association.
   ind = range(1, stop=Narray)  # [1:Narray].

   # Pick a starting coordinate at random.
   xs[:,1] = rand(D)
   jc, wc = nearestCells(xs[:,1],xb)
   cx[1] = jc[argmax(wc)]
   act[1] = true

   np = 1

   flg = true
   while (flg)

      # Pick a random one of the active points.
      iact = ind[act]
      nact = length(iact)

      if (nact == 0)
         flg = false
      else

         jp   = round(Int,(-0.5 + eps()) + nact*(rand() - eps()) + 1)
         ip   = iact[jp]

         # Sample k points about the hypersphere in sequence, looking for 
         # one that is at least a distance r from existing points.
         kflg = true
         kk = 0
         while (kflg)

            kk = kk + 1

            vec = -1.0 .+ 2.0*rand(D)
            lv = 0.0
            for jj in 1:D
               lv = lv + vec[jj]^2
            end
            n = vec/sqrt(lv)

            xtry = xs[:,ip] + (r + del) .* n

            # Wrap around the domain.
            for jj in 1:D
               if (xtry[jj] < 0.0)
                  xtry[jj] = xtry[jj] + 1.0
               elseif (xtry[jj] > 1.0)
                  xtry[jj] = xtry[jj] - 1.0
               end
            end

            # Compute the minimum distance to other points.
            jc, wc = nearestCells(xtry,xb)
            isur = surroundingCells(xtry,xb)
            Ncell = length(isur)
            dmin = 2.0*r
            for icell in 1:Ncell
               jcx = (cx .== isur[icell])
               if (sum(jcx) >= 1)  # There is a point associated with this cell.
                  cind = argmax(jcx)
                  dist = 0.0
                  for jj in 1:D
                     dist = dist + (xs[jj,cind] - xtry[jj])^2
                  end
                  dist = sqrt(dist)
                  if (dist < dmin)
                     dmin = dist
                  end
               end
            end

            if (dmin >= r)

               # Resize the arrays if needed.
               if (Narray == np)
                  Narray = 2*np   # Double on each resizing.

                  temp = deepcopy(act)
                  act = zeros(Bool,Narray)
                  act[1:np] = temp

                  temp = deepcopy(xs)
                  xs = zeros(D,Narray)
                  xs[:,1:np] = temp

                  temp = deepcopy(cx)
                  cx = zeros(Int,Narray)
                  cx[1:np] = temp

                  ind = range(1, stop=Narray)

               end
            
               # Success!  Log the point.
               np = np + 1
               xs[:,np] = xtry
               cx[np] = jc[argmax(wc)]
               act[np] = true
               kflg = false

            elseif (kk == k)

               act[ip] = false  # No available points around ip.
               kflg = false

            end

         end  # while (kflg)

#print("nact = ",nact,"  jp = ",jp,"  ip = ",ip,"  np = ",np,"\n")

      end  # if (nact != 0)

   end  # while (flg)

   return xs[:,1:np]

end  # function poissonDisk.

# ===================================================================
function selectPoints(xin, N)
#
# It is not possible to specify an exact number of points generated
# by the poissonDisk function.  The best that can be done is to
# generate a few too many points, and then take a subset of the
# space containing the desired number of points, rescaling to fill
# the unit cube.
#
# This works by performing a bisecting search along the diagonal
# from (0,0,...,0) to (1,1,...,1) until the proper number of points
# is encapsulated.  Then the coordinates are scaled up to the unit
# cube.
#
# Version:        Changes:
# --------        -------------
# 28.07.2021      Original code.
#
# Version:        Verification:
# --------        -------------
# 28.07.2021      Tested a 2D case.
#
# Inputs:
# -------
# xin             : Point coordinates defined on the unit cube.
#                   Array of dimension D-by-M.
# N               : Desired number of points on the unit cube.  Must
#                   be less than M.
#
# Outputs:
# --------
# x               : Coordinates of the N points normalized to fill
#                   the unit cube.

   D = size(xin,1)
   M = size(xin,2)

   # Declare variables that would otherwise be undefined outside the
   # "for" loop.  [Unintuitive Julia name scoping conventions...]
   ind = ones(Bool,M)
   a = 0.0

   if (N >= M)

      return xin

   else

      aLB = 0.0
      aUB = 1.0

      flg = true
      while (flg)

         # Find the bisecting point.
         a = 0.5*(aLB + aUB)

         # Find the number of points below the bisecting point.
         ind = ones(Bool,M)
         for jj in 1:D
            ind = (ind .& (xin[jj,:] .<= a))
         end
         nb = sum(ind)

         if (nb == N)
            flg = false
         elseif (nb > N)
            # Move down to eliminate points.
            aUB = a
         else  # nb < N
            # Move up to capture more points.
            aLB = a
         end

      end  # while (flg)

      return (xin[:,ind] * (1.0/a))

   end  # if (N >= M)

end  # selectPoints

# ===================================================================
function resample(xi, pri; Msam = 0)
#
# Based on the "systematic resampling" algorithm of Arulampalam 
# et al. A tutorial on particle filters for online nonlinear/
# non-Gaussian Bayesian tracking. IEEE Transactions on Signal
# Processing 50: 174-188.
#
# Version:        Changes:
# --------        -------------
# 16.08.2021      Original code.
#
# Version:        Verification:
# --------        -------------
# 16.08.2021      Checked using some straightforward cases.
#
# Inputs:
# -------
# xi              : D-by-N array of points.
# pri             : Input probabilities.
# Msam            : (Optional) Number of points to sample.
#
# Outputs:
# --------
# xo              : D-by-M array of resampled points.
# pro             : Output probabilities.

   D = size(xi,1)
   N = size(xi,2)

   if (Msam == 0)
      M = N
   else
      M = Msam
   end

   C = zeros(N)
   C[1] = pri[1]
   for ii in 2:N-1
      C[ii] = C[ii-1] + pri[ii]
   end
   C[N] = 1.0 + eps()

   Mm1 = 1.0/M

   u1 = rand()*Mm1

   xo = zeros(D,M)
   pro = zeros(M)
   io = zeros(Int,M)
   ii = 1
   for jj in 1:M
      u = u1 + (jj-1)*Mm1
      while (u > C[ii])
         ii = ii + 1
      end
      xo[:,jj] = xi[:,ii]
      pro[jj] = Mm1
      io[jj] = ii
   end

   return xo, pro, io

end  # function resample

end  # module OPMC.


