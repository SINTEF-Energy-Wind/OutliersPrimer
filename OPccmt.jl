module OPccmt
#
# Define generic functions used in cell-to-cell mapping of dynamic
# systems.
#

export nearestCells, surroundingCells, xFromCell, assignPhi

using SparseArrays

# ===================================================================
function nearestCells(x,xb)
#
# Return the nearest cell indices, for the cells surrounding each
# entry in a vector of states.  Also the weights for a linear
# interpolation.
#
# Version:        Changes:
# --------        -------------
# 26.03.2020      Original code.
# 27.07.2021      Streamlined the code.
#
# Version:        Verification:
# --------        -------------
# 26.03.2020      Verified that it reproduces the original Octave
#                 code, used in a variety of example problems.
# 27.07.2021      Verified that it reproduces the old nearestCells
#                 output.  Checked using @btime, about 20% faster.
#
# Inputs:
# -------
# x               : Ndim-by-Nvecs array containing the vectors.
# xb              : Ndim-by-3 array containing xmin, dx, xmax for each 
#                   dimension; that is, the possible values.
#
# Outputs:
# --------
# ic              : nearest cell ids, 2^Ndim-by-Nvecs.
# wc              : weights associated with each of the ic cells.

Ndim = size(x,1)
Nvecs = size(x,2)
Nx = round.(Int, (xb[:,3] - xb[:,1])./xb[:,2]) .+ 1

vx  = (x .- xb[:,1])./xb[:,2]
ixx = min.(max.(vx .+ 1,1),Nx)
ixL = min.(max.(floor.(Int,vx) .+ 1,1),Nx.-1)
wL  = 1 .- (ixx - ixL)

ic = zeros(Int,2^Ndim,Nvecs)
wc = zeros(2^Ndim,Nvecs)

for jj = 1:2^Ndim

   b = string(jj-1, base=2, pad=Ndim)
   ix = deepcopy(ixL)
   wx = deepcopy(wL)

   # The values that are 1 in binary, make "high".
   for kk in 1:Ndim
      hi = parse(Bool, b[kk])
      ix[kk,:] = ix[kk,:] .+ hi
      if (hi)
         wx[kk,:] = 1 .- wL[kk,:]
      end
   end

   wc[jj,:] = prod(wx, dims=1)

   for idim = 1:Ndim

      if (idim == Ndim)
         ic[jj,:] = ic[jj,:] + ix[idim,:]
      else
         ic[jj,:] = ic[jj,:] + prod(Nx[idim+1:Ndim]).*(ix[idim,:].-1)
      end

   end
end

ic = min.(max.(ic,1),prod(Nx))

return ic, wc

end # nearestCells

# ===================================================================
function surroundingCells(x,xb)
#
# Return the cells surrounding (adjacent to) the nearest cell.
#
# Version:        Changes:
# --------        -------------
# 27.07.2021      Original code.
#
# Version:        Verification:
# --------        -------------
# 27.07.2021      Checked a simple 2D problem, including out-of-bounds.
#
# Inputs:
# -------
# x               : Ndim-by-Nvecs array containing the vectors.
# xb              : Ndim-by-3 array containing xmin, dx, xmax for each
#                   dimension; that is, the possible values.
#
# Outputs:
# --------
# ic              : Surrounding cell ids, 3^Ndim-by-Nvecs.  In the case
#                   where the x coordinate is out of bounds, there may
#                   be repeated entries, but they are truncated to
#                   valid indices.

Ndim = size(x,1)
Nvecs = size(x,2)
Nx = round.(Int, (xb[:,3] - xb[:,1])./xb[:,2]) .+ 1
Ncel = 3^Ndim   # Ndim box of surrounding cells, including the center.

# First find the nearest cell.
jc, wc = nearestCells(x,xb)
jnear = zeros(Int,Nvecs)
for jj in 1:Nvecs
   imax = argmax(wc[:,jj])
   jnear[jj] = jc[imax,jj]
end

# Next identify the lower bound among the surrounding cells.
xnear = xFromCell(jnear,xb)
xlb = xnear .- 0.1*xb[:,2]  # Move a fraction of dx towards the LB.

# Find the lower-bound index on the surrounding cells.
vx  = (xlb .- xb[:,1])./xb[:,2]
ixx = min.(max.(vx .+ 1,1),Nx)
ixL = min.(max.(floor.(Int,vx) .+ 1,1),Nx.-1)

ic = zeros(Int, Ncel, Nvecs)

for jj in 1:Ncel

   # Base 3 number: 0 = "low", 1 = "mid", 2 = "high".
   b = string(jj-1, base=3, pad=Ndim)

   # Increment the values of ix according to the present cell ID.
   ix = deepcopy(ixL)
   for kk in 1:Ndim
      ix[kk,:] = ix[kk,:] .+ parse(Int, b[kk])
   end

   for kk = 1:Ndim

      if (kk == Ndim)
         ic[jj,:] = ic[jj,:] + ix[kk,:]
      else
         ic[jj,:] = ic[jj,:] + prod(Nx[kk+1:Ndim]).*(ix[kk,:].-1)
      end

   end

end

ic = min.(max.(ic,1),prod(Nx))

end  # surroundingCells

# ===================================================================
function xFromCell(ic,xb)
#
# Return the cell states for a given cell ID.
#
# Version:        Changes:
# --------        -------------
# 28.03.2020      Original code.
#
# Version:        Verification:
# --------        -------------
# 28.03.2020      Verified that it reproduces the original Octave
#                 code, used in a variety of example problems.
#
# Inputs:
# -------
# ic              : nearest cell id.
# xb              : N-by-3 array containing xmin, dx, xmax for each x;
#                   that is, the possible values.
#
# Outputs:
# --------
# x               : the vector of N states.

icr = deepcopy(ic)
Ndim = size(xb,1)
Nvec = size(ic,1)
Nx = round.(Int, (xb[:,3] - xb[:,1])./xb[:,2]) .+ 1
ix = zeros(Int,Ndim,Nvec)
for idim = 1:Ndim

   if (idim == Ndim)
      ix[idim,:] = icr
   else
      pr = prod(Nx[idim+1:Ndim])
      n = floor.(Int,icr./pr)
      rem = mod.(icr,pr)
      ind = (rem .== 0)
      n[ind] = n[ind] .- 1
      ix[idim,:] = n .+ 1
      icr = icr - pr.*n
   end

end

x = xb[:,1] .+ (ix .- 1).*xb[:,2]

return x

end # xFromCell

# ===================================================================
function assignPhi(jc,wc,Np)

Nj1 = size(jc,1)
Nnz = Nj1*size(jc,2)
ir = reshape(jc,Nnz)
ind = range(0,Nnz-Nj1+1,step=Nj1)
ic = zeros(Int,Nnz)
for irep = 1:Nj1
   ic[irep.+ind] = range(1,size(jc,2),step=1)
end
ss = reshape(wc,Nnz)
Phi = sparse(ir,ic,ss,Np,size(jc,2))
return Phi

end # assignPhi

end # module
