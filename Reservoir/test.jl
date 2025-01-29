using Test
using QuantumDots, QuantumDots.BlockDiagonals
using LinearAlgebra
using Random
using OrdinaryDiffEqTsit5
using LinearSolve
using Plots
using Statistics
using MLJLinearModels
using ExponentialUtilities
using KrylovKit
using UnPack
Random.seed!(1234)
includet("..\\system.jl")
includet("src.jl")
includet("narma.jl")
includet("training.jl")

