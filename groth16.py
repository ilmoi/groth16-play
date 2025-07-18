import galois
import numpy as np


# ----------------------------------------- R1CS

# use this for fast generation (but proof will fail)
# p = 71

# use this for correct generation (proof will pass) (but it's slow)
p = 21888242871839275222246405745257275088548364400416034343698204186575808495617
FP = galois.GF(p)

x = FP(2)
y = FP(3)

v1 = x * x
v2 = y * y
v3 = 5 * x * v1
v4 = 4 * v1 * v2
out = 5*x**3 - 4*x**2*y**2 + 13*x*y**2 + x**2 - 10*y

w = FP([1, out, x, y, v1, v2, v3, v4])

print("w =", w)

R = FP([[0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 5, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 4, 0, 0, 0],
         [0, 0, 13, 0, 0, 0, 0, 0]])

L = FP([[0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0]])

O = FP([[0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 1],
         [0, 1, 0, 10, FP(p - 1), 0, FP(p - 1), 1]])

Lw = np.dot(L, w)
Rw = np.dot(R, w)
Ow = np.dot(O, w)

print("Lw =", Lw)
print("Rw =", Rw)

LwRw = np.multiply(Lw, Rw)

print("Lw * Rw =", LwRw)

print("Ow =     ", Ow)

assert np.all(LwRw == Ow)

# ----------------------------------------- QAP

mtxs = [L, R, O]
poly_m = []

for m in mtxs:
    poly_list = []
    # for each column of the matrix
    for i in range(0, m.shape[1]):
        # create 2 arrays sized = number of rows of the matrix
        points_x = FP(np.zeros(m.shape[0], dtype=int))
        points_y = FP(np.zeros(m.shape[0], dtype=int))
        # for each row of the matrix
        for j in range(0, m.shape[0]):
            # x values are 1, 2, 3, 4, 5
            points_x[j] = FP(j+1)
            # y values are the values of the matrix
            points_y[j] = m[j][i]
            # so now you basically have a list of points (x, y)
            # and you want to find the lowest degree polynomial that passes through all the points
            # this is the Lagrange polynomial
            
        # find the lowest degree polynomial that passes through all the points
        poly = galois.lagrange_poly(points_x, points_y)
        # get the coefficients of the polynomial
        coef = poly.coefficients()[::-1]
        # if the polynomial is not of the same degree as the matrix, add zeros to the end
        # this is because the polynomial is of the form a_0 + a_1*x + a_2*x^2 + ... + a_n*x^n
        # and we want to make sure that the polynomial is of the same degree as the matrix
        
        # q: how is it possible that the polynomial is not of the same degree as the matrix?
        # because the matrix is not full rank, so there are some rows that are linearly dependent
        # so the polynomial is not of the same degree as the matrix
        # so we need to add zeros to the end of the polynomial
        # this is because the polynomial is of the form a_0 + a_1*x + a_2*x^2 + ... + a_n*x^n
        # and we want to make sure that the polynomial is of the same degree as the matrix
        if len(coef) < m.shape[0]:
            coef = np.append(coef, np.zeros(m.shape[0] - len(coef), dtype=int))
        poly_list.append(coef)
    
    poly_m.append(FP(poly_list))

Lp = poly_m[0]
Rp = poly_m[1]
Op = poly_m[2]

# this gives us matrices of the coefficients of the polynomials
# so now we have 3 matrices of the coefficients of the polynomials
# Lp, Rp, Op
# each matrix is of the form:
# [a_0, a_1, a_2, ..., a_n]
# [b_0, b_1, b_2, ..., b_n]
# [c_0, c_1, c_2, ..., c_n]

print(f'''L
{Lp}
''')

print(f'''R
{Rp}
''')

print(f'''O
{Op}
''')

# --------------------- tau

print("Setup phase")
print("-"*10)
print("Toxic waste:")
tau = FP(20)
# random secret value that must be destroyed during setup
# anyone who knows this value can forge proofs

print(f"τ = {tau}")

# --------------------- U, V, W

# multiply (dot product) each of the matrices by the witness vector w
# and then reverse the order of the coefficients
# this is because the coefficients are in the reverse order of the polynomial

# this means we went from 3 n x m matrices to 3 polynomials of degree m-1, one per matrix

U = galois.Poly((w @ Lp)[::-1])
V = galois.Poly((w @ Rp)[::-1])
W = galois.Poly((w @ Op)[::-1])

print("U = ", U)
print("V = ", V)
print("W = ", W)

# --------------------- T

# T is one of the 2 more polynomials we need to balance the equation

T = galois.Poly([1, p-1], field=FP)
for i in range(2, L.shape[0] + 1):
    T *= galois.Poly([1, p-i], field=FP) # this is the product of the polynomials
    # so T is the product of the polynomials
    # T = (1 - x) * (1 - 2x) * (1 - 3x) * ... * (1 - (p-1)x)
    # this is the product of the polynomials
    # T = (1 - x) * (1 - 2x) * (1 - 3x) * ... * (1 - (p-1)x)

# test how it behaves on different values of X
print("\nT = ", T)
for i in range(1, L.shape[0] + 2):
    print(f"T({i}) = ", T(i))
    if i == L.shape[0]:
        print("-"*10)

T_tau = T(tau)
print(f"\nT(τ) = {T_tau}")

# --------------------- H

# H is the other polynomial we need to balance the equation

H = (U * V - W) // T
rem = (U * V - W) % T

print("H = ", H)
print("rem = ", rem)

# check if the remainder is 0, this verifies if computaiton correct
assert rem == 0

u = U(tau)
v = V(tau)
_w = W(tau) #w takes for witness vector
ht = H(tau)*T_tau

# at this point we've boiled down R1CS into 5 polynomials
# These polynomials will serve as the foundation for the next chapter.
assert u * v - _w == ht, f"{u} * {v} - {_w} != {ht}"

# at this point we can evaluate the polynomials at tau
# but we have no assurances that:
# 1) Alice didn't invent these numbers and 
# 2) Alice is proving proof for the correct statement

# To solve this we need to place them into a common framework - EC.

# ----------------------------------------- EC

# first we'll do the NOT secure example, because it's easier to understand

# --------------------- trusted setup (NOT secure)

# we need a trusted setup because otherwise Alice could pick points where she knows the discrere logarithm

from py_ecc.optimized_bn128 import multiply, G1, G2, add, pairing, neg, normalize

# G1[τ^0], G1[τ^1], ..., G1[τ^d]
tau_G1 = [multiply(G1, int(tau**i)) for i in range(0, T.degree)]
# G1[τ^0 * T(τ)], G1[τ^1 * T(τ)], ..., G1[τ^d-1 * T(τ)]
target_G1 = [multiply(G1, int(tau**i * T_tau)) for i in range(0, T.degree - 1)]

# G2[τ^0], G2[τ^1], ..., G2[τ^d-1]
tau_G2 = [multiply(G2, int(tau**i)) for i in range(0, T.degree)]

print("Trusted setup:")
print("-"*10)
print(f"[τ]G1 = {[normalize(point) for point in tau_G1]}")
print(f"[T(τ)]G1 = {[normalize(point) for point in target_G1]}")

print(f"\n[τ]G2 = {[normalize(point) for point in tau_G2]}")

# --------------------- proof generation (NOT secure)

# returns a point on the curve that results from the dot product of the polynomial and the trusted points
def evaluate_poly(poly, trusted_points, verbose=False):
    coeff = poly.coefficients()[::-1]

    assert len(coeff) == len(trusted_points), "Polynomial degree mismatch!"

    if verbose:
        [print(normalize(point)) for point in trusted_points]

    # this is the stuff I implemented manually in the Bulletproof chapter
    terms = [multiply(point, int(coeff)) for point, coeff in zip(trusted_points, coeff)]
    # sum up, aka dot product
    evaluation = terms[0]
    for i in range(1, len(terms)):
        evaluation = add(evaluation, terms[i])

    if verbose:
        print("-"*10)
        print(normalize(evaluation))
    return evaluation

# we take each of the polynomials that we derived in the QAP chapter
# and we evaluate them at Tau, the point we created in the trusted setup

print("\nProof generation:")
print("-"*10)
# G1[u0 * τ^0] + G1[u1 * τ^1] + ... + G1[ud-1 * τ^d-1]
A_G1 = evaluate_poly(U, tau_G1)
# G2[v0 * τ^0] + G2[v1 * τ^1] + ... + G2[vd-1 * τ^d-1]
B_G2 = evaluate_poly(V, tau_G2)
# G1[w0 * τ^0] + G1[w1 * τ^1] + ... + G1[wd-1 * τ^d-1]
B_G1 = evaluate_poly(V, tau_G1)
# G1[w0 * τ^0] + G1[w1 * τ^1] + ... + G1[wd-1 * τ^d-1]
Cw_G1 = evaluate_poly(W, tau_G1)
# G1[h0 * τ^0 * T(τ)] + G1[h1 * τ^1 * T(τ)] + ... + G1[hd-2 * τ^d-2 * T(τ)]
HT_G1 = evaluate_poly(H, target_G1)

C_G1 = add(Cw_G1, HT_G1)

# so now we have 3 points that we'd be sharing with the prover

print(f"[A]G1 = {normalize(A_G1)}")
print(f"[B]G2 = {normalize(B_G2)}")
print(f"[C]G1 = {normalize(C_G1)}")

# --------------------- proof verification (NOT secure)

print("\nProof verification:")
print("-"*10)

# this pariding equation is the core of the Groth16 proof system
# e(A, B) == e(C, G2[1])
assert pairing(B_G2, A_G1) == pairing(G2, C_G1), "Pairing check failed!"
print("Pairing check passed!")

# this is NOT yet secure. Alice simply presents 3 points on the EC. 
# there isn't enough proof to show that actual computation took place and she didn't simply sub the params

# --------------------- proof verification (secure)

# lets add 2 more params - alpha and beta, which make sure Alice didn't tinker with the 2 points

# when we add 2 more params, we have to do:
# (A + alpha) * (B + beta) = alpha * beta + beta * A + alpha * B + C

# It’s crucial to note that the alpha and beta values must be encrypted by the setup agent, 
# similar to the tau value discussed in the previous article. Since both the prover and verifier are given alpha and beta, 
# leaving them unencrypted would still allow the prover to deceive the verifier.

# evalutates each polynomial in the list at the point x
def evaluate_poly_list(poly_list, x):
    results = []
    for poly in poly_list:
        results.append(poly(x))
    return results

def print_evaluation(name, results):
    print(f'\n{name} polynomial evaluations:')
    for i in range(0, len(results)):
        print(f'{name}_{i} = {results[i]}')

# it takes a matrix and returns a list of polynomials
# each polynomial is a row of the matrix
# each polynomial is a polynomial of the form a_0 + a_1*x + a_2*x^2 + ... + a_n*x^n
# and we want to make sure that the polynomial is of the same degree as the matrix
# so we reverse the coefficients of the polynomial
# and we return a list of polynomials
def to_poly(mtx):
    poly_list = []
    for i in range(0, mtx.shape[0]):
        poly_list.append( galois.Poly(mtx[i][::-1]) )
    return poly_list

def print_poly(name, poly_list):
    print(f'\n{name} polynomials:')
    for i in range(0, len(poly_list)):
        print(f'{name}_{i} = {poly_list[i]}')

# redo the setup, but this time we add 2 more params - alpha and beta
print("Setup phase")
print("-"*10)
print("Toxic waste:")
alpha = FP(2)
beta = FP(3)
tau = FP(20)

print(f"α = {alpha}")
print(f"β = {beta}")
print(f"τ = {tau}")

# multiply each of the matrices by alpha and beta
beta_L = beta * Lp
alpha_R = alpha * Rp
K = beta_L + alpha_R + Op # preimage of [βA + αB + C]

# this prints a bunch of 4th degree polynomials, not yet evaluated at tau
Kp = to_poly(K)
print_poly("K", Kp)

# this actually evaluates the polynomials at tau and prints scalars
print("K evaluations:")
K_eval = evaluate_poly_list(Kp, tau)
print([int(k) for k in K_eval])

# --------------------- trusted setup (secure)

# like last time, but this time have to also do 2 new params - alpha and beta

from py_ecc.optimized_bn128 import multiply, G1, G2, add, pairing, neg, normalize, eq

# G1[α]
alpha_G1 = multiply(G1, int(alpha))
# G1[β]
beta_G1 = multiply(G1, int(beta))
# G1[τ^0], G1[τ^1], ..., G1[τ^d]
tau_G1 = [multiply(G1, int(tau**i)) for i in range(0, T.degree)]
# G1[βU0(τ) + αV0(τ) + W0(τ)], G1[βU1(τ) + αV1(τ) + W1(τ)], ..., G1[βUd(τ) + αVd(τ) + Wd(τ)]
k_G1 = [multiply(G1, int(k)) for k in K_eval]
# G1[τ^0 * T(τ)], G1[τ^1 * T(τ)], ..., G1[τ^d-1 * T(τ)]
target_G1 = [multiply(G1, int(tau**i * T_tau)) for i in range(0, T.degree - 1)]

# G2[β]
beta_G2 = multiply(G2, int(beta))
# G2[τ^0], G2[τ^1], ..., G2[τ^d-1]
tau_G2 = [multiply(G2, int(tau**i)) for i in range(0, T.degree)]

print("Trusted setup:")
print("-"*10)
print(f"[α]G1 = {normalize(alpha_G1)}")
print(f"[β]G1 = {normalize(beta_G1)}")
print(f"[τ]G1 = {[normalize(point) for point in tau_G1]}")
print(f"[k]G1 = {[normalize(point) for point in k_G1]}")
print(f"[τT(τ)]G1 = {[normalize(point) for point in target_G1]}")

print(f"\n[β]G2 = {normalize(beta_G2)}")
print(f"[τ]G2 = {[normalize(point) for point in tau_G2]}")

# --------------------- proof generation (secure)

print("\nProof generation:")
print("-"*10)

U = galois.Poly((w @ Lp)[::-1])
V = galois.Poly((w @ Rp)[::-1])

# G1[u0 * τ^0] + G1[u1 * τ^1] + ... + G1[ud-1 * τ^d-1]
A_G1 = evaluate_poly(U, tau_G1)
# G1[A] = G1[A] + G1[α]
# NOTE: this is important!!!!! Vs the non secure example, we add alpha to A, not B
A_G1 = add(A_G1, alpha_G1)
# G2[v0 * τ^0] + G2[v1 * τ^1] + ... + G2[vd-1 * τ^d-1]
B_G2 = evaluate_poly(V, tau_G2)
# G2[B] = G2[B] + G2[β]
# NOTE: this is important!!!!! Vs the non secure example, we add beta to B, not A
B_G2 = add(B_G2, beta_G2)
# G1[h0 * τ^0 * T(τ)] + G1[h1 * τ^1 * T(τ)] + ... + G1[hd-2 * τ^d-2 * T(τ)]
HT_G1 = evaluate_poly(H, target_G1)
assert len(w) == len(k_G1), "Polynomial degree mismatch!"
# w0 * G1[k0] + w1 * G1[k1] + ... + wd-1 * G1[kd-1]
K_G1_terms = [multiply(point, int(scaler)) for point, scaler in zip(k_G1, w)]
K_G1 = K_G1_terms[0]
for i in range(1, len(K_G1_terms)):
    K_G1 = add(K_G1, K_G1_terms[i])

C_G1 = add(HT_G1, K_G1)

print(f"[A]G1 = {normalize(A_G1)}")
print(f"[B]G2 = {normalize(B_G2)}")
print(f"[C]G1 = {normalize(C_G1)}")
print("-" * 10)

# these are the 2 new params that the verifier uses to verify the proof, that make it secure
print("Verifier uses:")
print(f"[α]G1 = {normalize(alpha_G1)}")
print(f"[β]G1 = {normalize(beta_G1)}")

# --------------------- proof verification (secure)

# A = A + α
# B = B + β
# C = βA + αB + C
# AB == αβ + [βA + αB + C]
# TODO: doesn't work, not sure why
# assert pairing(B_G2, A_G1) == pairing(beta_G2, alpha_G1) + pairing(G2, C_G1)

# --------------------- proof verification (secure, using smart contract)

from string import Template

with open("Verifier.sol.template", "r") as f:
    template = Template(f.read())
    variables = {
        "aG1_x": normalize(neg(A_G1))[0],
        "aG1_y": normalize(neg(A_G1))[1],
        "bG2_x1": normalize(B_G2)[0].coeffs[0],
        "bG2_x2": normalize(B_G2)[0].coeffs[1],
        "bG2_y1": normalize(B_G2)[1].coeffs[0],
        "bG2_y2": normalize(B_G2)[1].coeffs[1],
        "cG1_x": normalize(C_G1)[0],
        "cG1_y": normalize(C_G1)[1],
        "alphaG1_x": normalize(alpha_G1)[0],
        "alphaG1_y": normalize(alpha_G1)[1],
        "betaG2_x1": normalize(beta_G2)[0].coeffs[0],
        "betaG2_x2": normalize(beta_G2)[0].coeffs[1],
        "betaG2_y1": normalize(beta_G2)[1].coeffs[0],
        "betaG2_y2": normalize(beta_G2)[1].coeffs[1],
    }
    output = template.substitute(variables)

with open("Verifier.sol", "w") as f:
    f.write(output)
    
print('yay it worked!');

# ----------------------------------------- public vs private polynomial

# ---------------------- do the split

# now we need to split the polynomials into public and private parts
# [1, out, x, y, v1, v2, v3, v4]
# first 2 are public, last 6 are private

def split_poly(poly):
    coef = [int(c) for c in poly.coefficients()]
    p1 = coef[-2:]
    p2 = coef[:-2] + [0] * 2

    return galois.Poly(p1, field=FP), galois.Poly(p2, field=FP)

u = U(tau)
v = V(tau)
_w = W(tau) # w taken by witness vector
ht = H(tau)*T_tau

U1, U2 = split_poly(U)
V1, V2 = split_poly(V)
W1, W2 = split_poly(W)

w1 = W1(tau)
w2 = W2(tau)

u1 = U1(tau)
u2 = U2(tau)

v1 = V1(tau)
v2 = V2(tau)

c = (beta * u2 + alpha * v2 + w2) + ht 
k = (beta * u1 + alpha * v1 + w1)

assert (u + alpha) * (v + beta) == alpha * beta + k + c # should be equal

# ---------------------- verifier computes K (public)

k_pub_G1, k_priv_G1 = k_G1[:2], k_G1[2:]
pub_input, priv_input = w[:2], w[2:]

print(f"[k_pub]G1 = {[normalize(point) for point in k_pub_G1]}")
print(f"[k_priv]G1 = {[normalize(point) for point in k_priv_G1]}")

print(f"pub_input = {pub_input}")
print(f"priv_input = {priv_input}")

# ---------------------- prover computes C (private)

K_priv_G1_terms = [multiply(point, int(scaler)) for point, scaler in zip(k_priv_G1, priv_input)]
K_priv_G1 = K_priv_G1_terms[0]
for i in range(1, len(K_priv_G1_terms)):
    K_priv_G1 = add(K_priv_G1, K_priv_G1_terms[i])

C_G1 = add(HT_G1, K_priv_G1)

print("\nProof generation:")
print("-"*10)
print(f"[A]G1 = {normalize(A_G1)}")
print(f"[B]G2 = {normalize(B_G2)}")
print(f"[C]G1 = {normalize(C_G1)}")
print("-" * 10)
print("Verifier uses:")
print(f"[α]G1 = {normalize(alpha_G1)}")
print(f"[β]G1 = {normalize(beta_G1)}")
print(f"[K terms]G1 = {[normalize(p) for p in k_pub_G1]}")

# ---------------------- verification

from string import Template

k1 = normalize(k_pub_G1[0])
k2 = normalize(k_pub_G1[1])

with open("VerifierPublicInput.sol.template", "r") as f:
    template = Template(f.read())
    variables = {
        "aG1_x": normalize(A_G1)[0],
        "aG1_y": normalize(A_G1)[1],
        "bG2_x1": normalize(B_G2)[0].coeffs[0],
        "bG2_x2": normalize(B_G2)[0].coeffs[1],
        "bG2_y1": normalize(B_G2)[1].coeffs[0],
        "bG2_y2": normalize(B_G2)[1].coeffs[1],
        "cG1_x": normalize(C_G1)[0],
        "cG1_y": normalize(C_G1)[1],
        "alphaG1_x": normalize(alpha_G1)[0],
        "alphaG1_y": normalize(alpha_G1)[1],
        "betaG2_x1": normalize(beta_G2)[0].coeffs[0],
        "betaG2_x2": normalize(beta_G2)[0].coeffs[1],
        "betaG2_y1": normalize(beta_G2)[1].coeffs[0],
        "betaG2_y2": normalize(beta_G2)[1].coeffs[1],
        "k1G1_x": k1[0],
        "k1G1_y": k1[1],
        "k2G1_x": k2[0],
        "k2G1_y": k2[1],
        "one": pub_input[0],
        "out": pub_input[1],
    }
    output = template.substitute(variables)

with open("VerifierPublicInput.sol", "w") as f:
    f.write(output)
    
# 2 more things to do:

# ---------------------------------------- make sure prover can't cheat

# make sure that the prover can't cheat by sending someone else's proof
# trusted party sends 2 more params - gamma and delta, and those are incorporated into the proof

print("Setup phase")
print("-"*10)
print("Toxic waste:")
alpha = FP(2)
beta = FP(3)
gamma = FP(4)
delta = FP(5)
tau = FP(20)

print(f"α = {alpha}")
print(f"β = {beta}")
print(f"γ = {gamma}")
print(f"δ = {delta}")
print(f"τ = {tau}")

def split_poly(poly):
    coef = [int(c) for c in poly.coefficients()]
    p1 = coef[-2:]
    p2 = coef[:-2] + [0] * 2

    return galois.Poly(p1, field=FP), galois.Poly(p2, field=FP)

u = U(tau)
v = V(tau)
ht = H(tau)*T_tau

U1, U2 = split_poly(U)
V1, V2 = split_poly(V)
W1, W2 = split_poly(W)

w1 = W1(tau)
w2 = W2(tau)

u1 = U1(tau)
u2 = U2(tau)

v1 = V1(tau)
v2 = V2(tau)

c = (beta * u2 + alpha * v2 + w2) * delta**-1 + ht * delta**-1
k = (beta * u1 + alpha * v1 + w1) * gamma**-1

a = u + alpha
b = v + beta

assert a * b == alpha * beta + k * gamma + c * delta # should be equal.

# ----------------- compute K (public)

print("Setup phase")
print("-"*10)
print("Toxic waste:")
alpha = FP(2)
beta = FP(3)
gamma = FP(4)
delta = FP(5)
tau = FP(20)

print(f"α = {alpha}")
print(f"β = {beta}")
print(f"γ = {gamma}")
print(f"δ = {delta}")
print(f"τ = {tau}")

# ----------------- setup phase

def split_poly(poly):
    coef = [int(c) for c in poly.coefficients()]
    p1 = coef[-2:]
    p2 = coef[:-2] + [0] * 2

    return galois.Poly(p1, field=FP), galois.Poly(p2, field=FP)

# U, V, W - witness already computed

u = U(tau)
v = V(tau)
_w = W(tau)
ht = H(tau)*T_tau

assert u * v == _w + ht

U1, U2 = split_poly(U)
V1, V2 = split_poly(V)
W1, W2 = split_poly(W)

w1 = W1(tau)
w2 = W2(tau)

u1 = U1(tau)
u2 = U2(tau)

v1 = V1(tau)
v2 = V2(tau)

c = (beta * u2 + alpha * v2 + w2) * delta**-1 + ht * delta**-1
k = (beta * u1 + alpha * v1 + w1) * gamma**-1

a = u + alpha
b = v + beta

assert a * b == alpha * beta + k * gamma + c * delta

alpha_G1 = multiply(G1, int(alpha))
beta_G2 = multiply(G2, int(beta))
gamma_G2 = multiply(G2, int(gamma))
delta_G2 = multiply(G2, int(delta))

tau_G1 = [multiply(G1, int(tau**i)) for i in range(0, T.degree)]
tau_G2 = [multiply(G2, int(tau**i)) for i in range(0, T.degree)]

powers_tauTtau_div_delta = [(tau**i * T_tau) / delta for i in range(0, T.degree - 1)]
target_G1 = [multiply(G1, int(pTd)) for pTd in powers_tauTtau_div_delta]

assert len(target_G1) == len(H.coefficients()), f"target_G1 length mismatch! {len(target_G1)} != {len(H.coefficients())}"

print("Trusted setup:")
print("-"*10)
print(f"[α]G1 = {normalize(alpha_G1)}")
print(f"[β]G2 = {normalize(beta_G2)}")
print(f"[γ]G2 = {normalize(gamma_G2)}")
print(f"[δ]G2 = {normalize(delta_G2)}")
print(f"[τ]G1 = {[normalize(point) for point in tau_G1]}")
print(f"[τ]G2 = {[normalize(point) for point in tau_G2]}")
print(f"[τT(τ)/δ]G1 = {[normalize(point) for point in target_G1]}")

# ----------------- compute K (public)

w_pub = w[:2]
w_priv = w[2:]

K_gamma, K_delta = [k/gamma for k in K_eval[:2]], [k/delta for k in K_eval[2:]]

print(f"K/γ = {[int(k) for k in K_gamma]}")
print(f"K/δ = {[int(k) for k in K_delta]}")

K_gamma_G1 = [multiply(G1, int(k)) for k in K_gamma]
K_delta_G1 = [multiply(G1, int(k)) for k in K_delta]

print(f"[K/γ]G1 = {[normalize(point) for point in K_gamma_G1]}")
print(f"[K/δ]G1 = {[normalize(point) for point in K_delta_G1]}")

# [K/γ*w]G1
Kw_gamma_G1_terms = [multiply(point, int(scaler)) for point, scaler in zip(K_gamma_G1, w_pub)]
Kw_gamma_G1 = Kw_gamma_G1_terms[0]
for i in range(1, len(Kw_gamma_G1_terms)):
    Kw_gamma_G1 = add(Kw_gamma_G1, Kw_gamma_G1_terms[i])

print(f"[K/γ*w]G1 = {normalize(Kw_gamma_G1)}")

# [K/δ*w]G1
Kw_delta_G1_terms = [multiply(point, int(scaler)) for point, scaler in zip(K_delta_G1, w_priv)]
Kw_delta_G1 = Kw_delta_G1_terms[0]
for i in range(1, len(Kw_delta_G1_terms)):
    Kw_delta_G1 = add(Kw_delta_G1, Kw_delta_G1_terms[i])

print(f"[K/δ*w]G1 = {normalize(Kw_delta_G1)}")

# ----------------- compute HT

HT_G1 = evaluate_poly(H, target_G1)
print(f"\n[τT(τ)/δ]G1 = {normalize(HT_G1)}")

assert pairing(G2, multiply(G1, int(ht/delta))) == pairing(G2, HT_G1)

# ----------------- compute C (private)

C_G1 = add(Kw_delta_G1, HT_G1)

# ----------------- verify proof

K_G1 = Kw_gamma_G1

A_G1 = evaluate_poly(U, tau_G1)
A_G1 = add(A_G1, alpha_G1)
B_G2 = evaluate_poly(V, tau_G2)
B_G2 = add(B_G2, beta_G2)

k1 = normalize(K_gamma_G1[0])
k2 = normalize(K_gamma_G1[1])

with open("VerifierPublicInputGammaDelta.sol.template", "r") as f:
    template = Template(f.read())
    variables = {
        "aG1_x": normalize(A_G1)[0],
        "aG1_y": normalize(A_G1)[1],
        "bG2_x1": normalize(B_G2)[0].coeffs[0],
        "bG2_x2": normalize(B_G2)[0].coeffs[1],
        "bG2_y1": normalize(B_G2)[1].coeffs[0],
        "bG2_y2": normalize(B_G2)[1].coeffs[1],
        "cG1_x": normalize(C_G1)[0],
        "cG1_y": normalize(C_G1)[1],
        "alphaG1_x": normalize(alpha_G1)[0],
        "alphaG1_y": normalize(alpha_G1)[1],
        "betaG2_x1": normalize(beta_G2)[0].coeffs[0],
        "betaG2_x2": normalize(beta_G2)[0].coeffs[1],
        "betaG2_y1": normalize(beta_G2)[1].coeffs[0],
        "betaG2_y2": normalize(beta_G2)[1].coeffs[1],
        "k1G1_x": k1[0],
        "k1G1_y": k1[1],
        "k2G1_x": k2[0],
        "k2G1_y": k2[1],
        "one": pub_input[0],
        "out": pub_input[1],
        "gammaG2_x1": normalize(gamma_G2)[0].coeffs[0],
        "gammaG2_x2": normalize(gamma_G2)[0].coeffs[1],
        "gammaG2_y1": normalize(gamma_G2)[1].coeffs[0],
        "gammaG2_y2": normalize(gamma_G2)[1].coeffs[1],
        "deltaG2_x1": normalize(delta_G2)[0].coeffs[0],
        "deltaG2_x2": normalize(delta_G2)[0].coeffs[1],
        "deltaG2_y1": normalize(delta_G2)[1].coeffs[0],
        "deltaG2_y2": normalize(delta_G2)[1].coeffs[1],
    }
    output = template.substitute(variables)

with open("VerifierPublicInputGammaDelta.sol", "w") as f:
    f.write(output)

# ---------------------------------------- make sure verifier can't cheat

# make sure that the verifier can't cheat by re-using someone else's proof
# introduce randomness so that 2 of the same proofs actually look different

# ----------------- setup phase

print("Additional Trusted setup:")
print("-"*10)
delta_G1 = multiply(G1, int(delta))
gamma_G1 = multiply(G1, int(gamma))

print(f"[γ]G1 = {normalize(gamma_G1)}")
print(f"[δ]G1 = {normalize(delta_G1)}")

print("Prover picks random r, s")
r = FP(12)
s = FP(13)

print(f"r = {r}")
print(f"s = {s}")

def split_poly(poly):
    coef = [int(c) for c in poly.coefficients()]
    p1 = coef[-2:]
    p2 = coef[:-2] + [0] * 2

    return galois.Poly(p1, field=FP), galois.Poly(p2, field=FP)

# U, V, W - witness already computed

u = U(tau)
v = V(tau)
_w = W(tau)
ht = H(tau)*T_tau

assert u * v == _w + ht

U1, U2 = split_poly(U)
V1, V2 = split_poly(V)
W1, W2 = split_poly(W)

w1 = W1(tau)
w2 = W2(tau)

u1 = U1(tau)
u2 = U2(tau)

v1 = V1(tau)
v2 = V2(tau)

a = u + alpha + r * delta
b = v + beta + s * delta

c = ((beta * u2 + alpha * v2 + w2) * delta**-1 + ht * delta**-1) + s * a + r * b - r * s * delta
k = (beta * u1 + alpha * v1 + w1) * gamma**-1

assert a * b == alpha * beta + k * gamma + c * delta

# ----------------- prover recompute A B C

r_delta_G1 = multiply(delta_G1, int(r))
s_delta_G1 = multiply(delta_G1, int(s))
s_delta_G2 = multiply(delta_G2, int(s))

A_G1 = evaluate_poly(U, tau_G1)
A_G1 = add(A_G1, alpha_G1)
A_G1 = add(A_G1, r_delta_G1)

B_G2 = evaluate_poly(V, tau_G2)
B_G2 = add(B_G2, beta_G2)
B_G2 = add(B_G2, s_delta_G2)

B_G1 = evaluate_poly(V, tau_G1)
B_G1 = add(B_G1, beta_G1)
B_G1 = add(B_G1, s_delta_G1)

As_G1 = multiply(A_G1, int(s))
Br_G1 = multiply(B_G1, int(r))
rs_delta_G1 = multiply(delta_G1, int(-r*s))

C_G1 = add(Kw_delta_G1, HT_G1)
C_G1 = add(C_G1, As_G1)
C_G1 = add(C_G1, Br_G1)
C_G1 = add(C_G1, rs_delta_G1)

print(f"[A]G1 = {normalize(A_G1)}")
print(f"[B]G2 = {normalize(B_G2)}")
print(f"[C]G1 = {normalize(C_G1)}")

# ----------------- verify proof

k1 = normalize(K_gamma_G1[0])
k2 = normalize(K_gamma_G1[1])

with open("VerifierPublicInputGammaDelta.sol.template", "r") as f:
    template = Template(f.read())
    variables = {
        "aG1_x": normalize(A_G1)[0],
        "aG1_y": normalize(A_G1)[1],
        "bG2_x1": normalize(B_G2)[0].coeffs[0],
        "bG2_x2": normalize(B_G2)[0].coeffs[1],
        "bG2_y1": normalize(B_G2)[1].coeffs[0],
        "bG2_y2": normalize(B_G2)[1].coeffs[1],
        "cG1_x": normalize(C_G1)[0],
        "cG1_y": normalize(C_G1)[1],
        "alphaG1_x": normalize(alpha_G1)[0],
        "alphaG1_y": normalize(alpha_G1)[1],
        "betaG2_x1": normalize(beta_G2)[0].coeffs[0],
        "betaG2_x2": normalize(beta_G2)[0].coeffs[1],
        "betaG2_y1": normalize(beta_G2)[1].coeffs[0],
        "betaG2_y2": normalize(beta_G2)[1].coeffs[1],
        "k1G1_x": k1[0],
        "k1G1_y": k1[1],
        "k2G1_x": k2[0],
        "k2G1_y": k2[1],
        "one": pub_input[0],
        "out": pub_input[1],
        "gammaG2_x1": normalize(gamma_G2)[0].coeffs[0],
        "gammaG2_x2": normalize(gamma_G2)[0].coeffs[1],
        "gammaG2_y1": normalize(gamma_G2)[1].coeffs[0],
        "gammaG2_y2": normalize(gamma_G2)[1].coeffs[1],
        "deltaG2_x1": normalize(delta_G2)[0].coeffs[0],
        "deltaG2_x2": normalize(delta_G2)[0].coeffs[1],
        "deltaG2_y1": normalize(delta_G2)[1].coeffs[0],
        "deltaG2_y2": normalize(delta_G2)[1].coeffs[1],
    }
    output = template.substitute(variables)

with open("VerifierPublicInputGammaDeltaRS.sol", "w") as f:
    f.write(output)
    
print("Done!")