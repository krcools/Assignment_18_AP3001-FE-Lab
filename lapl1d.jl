using Pkg
Pkg.activate(".")
Pkg.instantiate()

using LinearAlgebra

struct Mesh
    vertices
    elements
end

function generatemesh(a, b, n)

    vertices = [[x] for x in range(a,b,length=n+1)]
    elements = [[i,i+1] for i in 1:n]

    return Mesh(vertices, elements)
end


function elementmatrix(mesh, element)
    i1 = element[1]
    i2 = element[2]
    v1 = mesh.vertices[i1]
    v2 = mesh.vertices[i2]
    el_size = norm(v2-v1)
    h1 = -1/el_size
    h2 = +1/el_size
    S = el_size * [
        h1*h1 h1*h2
        h2*h1 h2*h2]
    return S
end

function assemblematrix(mesh)
    T = Float64
    n = length(mesh.vertices)
    S = zeros(T,n,n)
    for element in mesh.elements
        Sel = elementmatrix(mesh, element)
        for (p,i) in enumerate(element)
            for (q,j) in enumerate(element)
                S[i,j] += Sel[p,q]
            end
        end
    end

    S = S[2:end-1,2:end-1]
end

function elementvector(f, mesh, element)
    i1 = element[1]
    i2 = element[2]
    v1 = mesh.vertices[i1]
    v2 = mesh.vertices[i2]
    el_size = norm(v2-v1)
    F = el_size * [
        0.5*f(v1)
        0.5*f(v2)]
    return F
end

function assemblevector(f, mesh)

    T = Float64
    n = length(mesh.vertices)
    F = zeros(T,n)
    for element in mesh.elements
        Fel = elementvector(f,mesh,element)
        for (p,i) in enumerate(element)
            F[i] += Fel[p]
        end
    end

    return F[2:end-1]
end

a, b, n_el = 0.0, 1.0, 20
mesh = generatemesh(0.0, 1.0, n_el)
S = assemblematrix(mesh)

# The first arg defines the function f as the function
# that takes on the value 1 in every point.
F = assemblevector(p -> 1.0, mesh)

u = S \ F

using Makie
x_axis = range(a, b, length=n_el+1)[2:end-1]
plot(x_axis, u)

u_exact(x) = -1/2*x*(x-1)
scatter!(x_axis, u_exact.(x_axis))