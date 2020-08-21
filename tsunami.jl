using Pkg
Pkg.activate(".")
Pkg.instantiate()

using CompScienceMeshes
using LinearAlgebra

fn = joinpath(@__DIR__, "assets", "world.msh")
border = CompScienceMeshes.read_gmsh_mesh(fn, physical="Border", dimension=1)
coast  = CompScienceMeshes.read_gmsh_mesh(fn, physical="Coast", dimension=1)
sea    = CompScienceMeshes.read_gmsh_mesh(fn, physical="Sea", dimension=2)

border_vertices = skeleton(border, 0)
coast_vertices  = skeleton(coast, 0)
sea_vertices    = skeleton(sea, 0)

interior_vertices = submesh(sea_vertices) do v
    v in border_vertices && return false
    v in coast_vertices && return false
    return true
end

border_indices = sort(unique(reduce(vcat, cells(border_vertices))))
coast_indices = sort(unique(reduce(vcat, cells(coast_vertices))))
interior_indices = sort(unique(reduce(vcat, cells(interior_vertices))))

function elementmatrix(mesh, element)
    v1 = mesh.vertices[element[1]]
    v2 = mesh.vertices[element[2]]
    v3 = mesh.vertices[element[3]]
    tangent1 = v3 - v2
    tangent2 = v1 - v3
    tangent3 = v2 - v1
    normal = (v1-v3) × (v2-v3)
    area = 0.5 * norm(normal)
    grad1 = (normal × tangent1) / (2 *area)
    grad2 = (normal × tangent2) / (2 *area)
    grad3 = (normal × tangent3) / (2 *area)

    S = area * [
        dot(grad1,grad1) dot(grad1,grad2) dot(grad1,grad3)
        dot(grad2,grad1) dot(grad2,grad2) dot(grad2,grad3)
        dot(grad3,grad1) dot(grad3,grad2) dot(grad3,grad3)]
end

function assemblematrix(mesh, active_indices)
    n = length(active_indices)
    S = zeros(n,n)
    for element in mesh
        Sel = elementmatrix(mesh, element)
        for (p,i) in enumerate(element)
            m = findfirst(==(i), active_indices)
            m == nothing && continue
            for (q,j) in enumerate(element)
                n = findfirst(==(j), active_indices)
                n == nothing && continue
                S[m,n] += Sel[p,q]
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


function assemblevector(f, mesh, active_indices)

    n = length(active_indices)
    F = zeros(n)
    for element in mesh
        Fel = elementvector(f,mesh,element)
        for (p,i) in enumerate(element)
            m = findfirst(==(i), active_indices)
            m == nothing && continue
            F[m] += Fel[p]
        end
    end

    return F
end

S = assemblematrix(sea, interior_indices)
F = assemblevector(p -> 1.0, sea, interior_indices)
u = S \ F

u_tilda = zeros(length(sea_vertices))
u_tilda[interior_indices] = u

using Makie
Makie.mesh(vertexarray(sea), cellarray(sea), color=u_tilda)