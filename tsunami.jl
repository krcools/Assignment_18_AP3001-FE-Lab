using Pkg
Pkg.activate(".")
Pkg.instantiate()

using CompScienceMeshes
using LinearAlgebra
using SparseArrays

fn = joinpath(@__DIR__, "assets", "world.msh")
border = CompScienceMeshes.read_gmsh_mesh(fn, physical="Border", dimension=1)
coast  = CompScienceMeshes.read_gmsh_mesh(fn, physical="Coast", dimension=1)
sea    = CompScienceMeshes.read_gmsh_mesh(fn, physical="Sea", dimension=2)

# skeleton creates lower dimensional meshes from a given mesh. With second argument
# zero the returned mesh is simply the cloud of vertices on which the original mesh
# was built.
border_vertices = skeleton(border, 0)
coast_vertices  = skeleton(coast, 0)
sea_vertices    = skeleton(sea, 0)

# The FEM as presented here solved the homogenous Dirichlet problem for the Laplace
# equations. This means that we will not be associating basis functions with vertices
# on either boundary. After filtering out these vertices we are left with only
# interior vertices.
interior_vertices = submesh(sea_vertices) do v
    v in border_vertices && return false
    v in coast_vertices && return false
    return true
end

"""
Creates the local to global map for FEM assembly.

    localtoglobal(active_vertices, domain) -> gl

The returned map `gl` can be called as

    gl(k,p)

Here, `k` is an index into `domain` (i.e. it refers to a specific element, and
`p` is a local index into a specific element. It ranges from 1 to 3 for triangular
elements and from 1 to 2 for segments. The function returns an index `i` into
`active_vertices` if the i-th active vertex equals the p-th vertex of element k and
`gl` return `nothing` otherwise.
"""
function localtoglobal(active_vertices, domain)
    conn = copy(transpose(connectivity(active_vertices, domain, abs)))
    nz = nonzeros(conn)
    rv = rowvals(conn)
    function gl(k,p)
        for q in nzrange(conn,k)
            nz[q] == p && return rv[q]
        end
        return nothing
    end
    return gl
end


function elementmatrix(mesh, element)
    v1 = mesh.vertices[element[1]]
    v2 = mesh.vertices[element[2]]
    v3 = mesh.vertices[element[3]]
    tangent1 = v3 - v2
    tangent2 = v1 - v3
    tangent3 = v2 - v1
    normal = (v1-v3) × (v2-v3)
    area = 0.5 * norm(normal)
    normal = normalize(normal)
    grad1 = (normal × tangent1) / (2 *area)
    grad2 = (normal × tangent2) / (2 *area)
    grad3 = (normal × tangent3) / (2 *area)

    S = area * [
        dot(grad1,grad1) dot(grad1,grad2) dot(grad1,grad3)
        dot(grad2,grad1) dot(grad2,grad2) dot(grad2,grad3)
        dot(grad3,grad1) dot(grad3,grad2) dot(grad3,grad3)]
end

function assemblematrix(mesh, active_vertices)
    n = length(active_vertices)
    S = zeros(n,n)
    gl = localtoglobal(active_vertices, mesh)
    for (k,element) in enumerate(mesh)
        Sel = elementmatrix(mesh, element)
        for p in 1:3
            i = gl(k,p)
            i == nothing && continue
            for q in 1:3
                j = gl(k,q)
                j == nothing && continue
                S[i,j] += Sel[p,q]
            end
        end
    end

    return S
end


function elementvector(f, mesh, element)
    v1 = mesh.vertices[element[1]]
    v2 = mesh.vertices[element[2]]
    v3 = mesh.vertices[element[3]]
    el_size = norm((v1-v3)×(v2-v3))/2
    F = el_size * [
        f(v1)/3
        f(v2)/3
        f(v3)/3]
    return F
end


function assemblevector(f, mesh, active_vertices)
    n = length(active_vertices)
    F = zeros(n)
    gl = localtoglobal(active_vertices, mesh)
    for (k,element) in enumerate(mesh)
        Fel = elementvector(f,mesh,element)
        for p in 1:3
            i = gl(k,p)
            i == nothing && continue
            F[i] += Fel[p]
        end
    end

    return F
end

# For the assignment of the lab, i.e. the Helmholtz equations (aka the frequency
# domain wave equation), subject to absorbing boundary conditions, you will also
# have to include a term stemming from boundary integral contributions. For that
# term a different local-to-global matrix is required: one linking segments on the
# boundary to indices of active vertices. You can create this map using the same
# function, i.e. like:
#
#   gl = localtoglobal(active_vertices, border)
#

S = assemblematrix(sea, interior_vertices)
F = assemblevector(p -> 1.0, sea, interior_vertices)
u = S \ F

u_tilda = zeros(length(sea_vertices))
for (j,m) in enumerate(interior_vertices)
    u_tilda[m[1]] = u[j]
end

using Makie
Makie.mesh(vertexarray(sea), cellarray(sea), color=u_tilda)