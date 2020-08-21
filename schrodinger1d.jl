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

function el_int(r1,r2,a,b,k)
    (r2^(k+3)-r1^(k+3))/(k+3) -
        (a+b)*(r2^(k+2)-r1^(k+2))/(k+2) +
        a*b*(r2^(k+1) - r1^(k+1))/(k+1)
end

function elementmatrix_lhs(mesh, element)
    i1 = element[1]
    i2 = element[2]
    v1 = mesh.vertices[i1]
    v2 = mesh.vertices[i2]

    r1 = v1[1]
    r2 = v2[1]

    el_size = norm(v2-v1)

    S1 = 1/2 * 1/3 * (r2^3 - r1^3) / el_size^2 * [
        1 -1
        -1 1]

    l = 0
    S2 = 1/2 * l*(l+1) * el_size / 6 * [
        2 1
        1 2]

    S3 = -1 / el_size^2 * [
        el_int(r1,r2,r1,r1,1) -el_int(r1,r2,r1,r2,1)
        -el_int(r1,r2,r2,r1,1) el_int(r1,r2,r2,r2,1)]

    return S1 + S2 + S3
end


function elementmatrix_rhs(mesh, element)
    i1 = element[1]
    i2 = element[2]
    v1 = mesh.vertices[i1]
    v2 = mesh.vertices[i2]

    r1 = v1[1]
    r2 = v2[1]

    el_size = norm(v2-v1)
    return 1 / el_size^2 * [
        el_int(r1,r2,r1,r1,2) -el_int(r1,r2,r1,r2,2)
        -el_int(r1,r2,r2,r1,2) el_int(r1,r2,r2,r2,2)]
end


function assemblematrix(elementmatrix, mesh)
    n = length(mesh.vertices)
    S = zeros(n,n)
    for element in mesh.elements
        Sel = elementmatrix(mesh, element)
        for (p,i) in enumerate(element)
            for (q,j) in enumerate(element)
                S[i,j] += Sel[p,q]
            end
        end
    end

    return S
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

a, b, n_el = 0.0, 100.0, 2000
mesh = generatemesh(a, b, n_el)
Sl = assemblematrix(elementmatrix_lhs, mesh)
Sr = assemblematrix(elementmatrix_rhs, mesh)

EV = eigen(Sl,Sr)

# The first arg defines the function f as the function
# that takes on the value 1 in every point.
# F = assemblevector(p -> 1.0, mesh)

# u = S \ F

# using Makie
# x_axis = range(a, b, length=n_el+1)[2:end-1]
# plot(x_axis, u)

# u_exact(x) = -1/2*x*(x-1)
# scatter!(x_axis, u_exact.(x_axis))

mass_el = 9.109e-31
h_bar = 6.6261e-34 / 2 / pi
charge_el = 1.602e-19
eps0 = 8.854e-12
bohr_radius = 4 * pi * eps0 * h_bar^2 / mass_el / charge_el^2

@show EV.values[1] / charge_el * h_bar^2 / mass_el / bohr_radius^2