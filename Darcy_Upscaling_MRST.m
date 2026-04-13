%% Probabilistic upscaling with MRST for a single fracture model
% Computes:
% 1) Darcy-upscaled permeability of the local Cubic-Law field (Biased)
% 2) Darcy-upscaled permeability of the descriptor fields:
%       lower, mode, mean, upper (Stokes compatible)
% 3) Optional Monte Carlo propagation from predicted log-normal parameters
%       mu, sigma
% Required inputs:
% - aperture file containing a_m
% - descriptor file containing lower, mode, mean, upper
% - mu/sigma file containing mu, sigma
%
% Notes:
% - Aperture and permeability maps are assumed to be stored in micrometers
%   and micrometer^2, respectively.
% - MRST solves in SI units, so permeability is converted to m^2.

tStart = cputime;
mrstModule add incomp
gravity off

%% ------------------------------------------------------------------------
% User settings
% -------------------------------------------------------------------------
% Grid size in pixels
nx = 1480;
ny = 1873;

% Physical voxel size
voxel_size_m = 2.75e-6;

% Flow direction: pressure increases from LEFT to RIGHT, flow is RIGHT to LEFT
pressure_gradient = 10;  % Pa.m^-1

% Optional Monte Carlo
run_monte_carlo = false;
Nmc = 400;
rng_seed = 1;

% Numerical safeguards
Kmin_m2 = 1e-20;

% Input files
frac_name = 'NaturalFrac_1';
aperture_file = sprintf('./outputs/Apertures_%s.mat', frac_name);                % contain a_m
descriptor_file = sprintf('./outputs/95CI_descriptors_%s.mat', frac_name);       % contain lower, mode, mean, upper
musigma_file = sprintf('./outputs/95CI_mu_sigma_params_%s.mat', frac_name);      % contain mu, sigma

% Optional plotting
plot_pressure_field = false;
plot_velocity_field = false;

%% ------------------------------------------------------------------------
% Geometry
% -------------------------------------------------------------------------
fprintf('Define geometry\n')

Lx = nx * voxel_size_m;
Ly = ny * voxel_size_m;

G = cartGrid([nx, ny], [Lx, Ly]);
G = computeGeometry(G);

%% ------------------------------------------------------------------------
% Fluid and boundary conditions
% -------------------------------------------------------------------------
fprintf('Set fluid and boundary conditions\n')

fluid = initSingleFluid( ...
    'mu',  1*centi*poise, ...
    'rho', 1014*kilogram/meter^3);

mu_visc = 1*centi*poise;

bc = pside([], G, 'LEFT', 0);
bc = pside(bc, G, 'RIGHT', pressure_gradient * Lx);

x = G.faces.centroids(:,1);
tol = 1e-12 * range(x);
facesLeft  = find(abs(x - min(x)) < tol);
facesRight = find(abs(x - max(x)) < tol);

facesOut = facesLeft;
dpdx = (pressure_gradient * Lx) / (max(x) - min(x));

%% ------------------------------------------------------------------------
% Load data
% -------------------------------------------------------------------------
fprintf('Load aperture and probabilistic permeability data\n')

aperture_data   = load(aperture_file, 'a_m');
descriptor_data = load(descriptor_file, 'lower', 'mode', 'mean', 'upper');
param_data      = load(musigma_file, 'mu', 'sigma');

a_m     = double(aperture_data.a_m)';
K_lower = double(descriptor_data.lower)';
K_mode  = double(descriptor_data.mode)';
K_mean  = double(descriptor_data.mean)';
K_upper = double(descriptor_data.upper)';

muHat    = double(param_data.mu)';
sigmaHat = double(param_data.sigma)';
sigmaHat = max(sigmaHat, 0);

% Transpose maps if needed to match MRST grid ordering
% [a_m, K_lower, K_mode, K_mean, K_upper, muHat, sigmaHat] = ...
%     ensure_map_orientation(a_m, K_lower, K_mode, K_mean, K_upper, muHat, sigmaHat, nx, ny);

%% ------------------------------------------------------------------------
% Darcy upscaling 
% -------------------------------------------------------------------------
fprintf('Compute Darcy upscaling\n')

% Local Cubic-Law permeability from mechanical aperture
K_LCL = a_m.^2 / 12;  % [um^2]
% Classical Cubic Law permeability
Keff_cubic_law_m2 = mean(a_m(:))^2/12 *1e-12; % [m^2]

results = struct();

results.Keff_cubic_law = local_computeKeff( ...
    G, K_LCL(:) * 1e-12, fluid, mu_visc, bc, facesOut, dpdx, Kmin_m2);

results.Keff_lower = local_computeKeff( ...
    G, K_lower(:) * 1e-12, fluid, mu_visc, bc, facesOut, dpdx, Kmin_m2);

results.Keff_mode = local_computeKeff( ...
    G, K_mode(:) * 1e-12, fluid, mu_visc, bc, facesOut, dpdx, Kmin_m2);

results.Keff_mean = local_computeKeff( ...
    G, K_mean(:) * 1e-12, fluid, mu_visc, bc, facesOut, dpdx, Kmin_m2);

results.Keff_upper = local_computeKeff( ...
    G, K_upper(:) * 1e-12, fluid, mu_visc, bc, facesOut, dpdx, Kmin_m2);

% Plain scalar outputs for easy reading in Python
Keff_local_cubic_law_m2 = results.Keff_cubic_law;
Keff_lower_m2     = results.Keff_lower;
Keff_mode_m2      = results.Keff_mode;
Keff_mean_m2      = results.Keff_mean;
Keff_upper_m2     = results.Keff_upper;

fprintf('\nUpscaled permeabilities [um^2]\n')
fprintf('  Cubic Law : %g\n', Keff_cubic_law_m2 *1e12)
fprintf('  Local Cubic Law : %g\n', Keff_local_cubic_law_m2 * 1e12)
fprintf('  Lower     : %g\n', Keff_lower_m2 * 1e12)
fprintf('  Mode      : %g\n', Keff_mode_m2 * 1e12)
fprintf('  Mean      : %g\n', Keff_mean_m2 * 1e12)
fprintf('  Upper     : %g\n', Keff_upper_m2 * 1e12)

%% ------------------------------------------------------------------------
% Optional: inspect pressure / velocity for the mean permeability map
% -------------------------------------------------------------------------

if plot_pressure_field || plot_velocity_field
    [resSol_mean, rock_mean] = solve_darcy( ...
        G, K_mean(:) * 1e-12, fluid, bc, Kmin_m2);

    if plot_pressure_field
        figure;
        plotCellData(G, resSol_mean.pressure(1:G.cells.num) * 1e3, ...
            'EdgeColor', 'k', 'EdgeAlpha', 0.05);
        title({'Cell Pressure [mPa]', 'from mean permeability map'})
        colorbar; axis equal tight; view(2); colormap jet
        axis off
        set(gca, 'YDir', 'reverse');
    end

    v_cell = faceFlux2cellVelocity(G, resSol_mean.flux);
    speed  = sqrt(sum(v_cell.^2, 2));

    if plot_velocity_field
        figure;
        plotCellData(G, speed * 1e6, 'EdgeColor', 'none');
        axis equal tight; view(2); colorbar
        title('Darcy velocity magnitude [\mum/s]')
        axis off
        set(gca, 'YDir', 'reverse');
    end
end

%% ------------------------------------------------------------------------
% Optional Monte Carlo propagation
% -------------------------------------------------------------------------
Keff_MonteCarlo_m2 = [];

if run_monte_carlo
    fprintf('\nRun Monte Carlo propagation\n')

    rng(rng_seed);

    muVec  = muHat(:);
    sigVec = sigmaHat(:);

    Keff_MonteCarlo_m2 = zeros(Nmc, 1);

    Kmax_m2 = quantile(K_mean(:) * 1e-12, 0.9999);

    for n = 1:Nmc
        % Fully correlated latent perturbation: one scalar z for the whole field
        z = randn();

        K_um2 = exp(muVec + sigVec * z);
        K_m2  = K_um2 * 1e-12;
        K_m2  = max(K_m2, Kmin_m2);
        K_m2  = min(K_m2, Kmax_m2);

        Keff_MonteCarlo_m2(n) = local_computeKeff( ...
            G, K_m2, fluid, mu_visc, bc, facesOut, dpdx, Kmin_m2);

        fprintf('MC %d/%d: Keff = %g m^2\n', n, Nmc, Keff_MonteCarlo_m2(n));
    end

    q = [0.025 0.5 0.975];
    q_mc = quantile(Keff_MonteCarlo_m2, q);

    results.Keff_mc = Keff_MonteCarlo_m2;
    results.Keff_mc_q025 = q_mc(1);
    results.Keff_mc_q50  = q_mc(2);
    results.Keff_mc_q975 = q_mc(3);

    fprintf('\nMonte Carlo quantiles [um^2]\n')
    fprintf('  Q0.025 : %g\n', q_mc(1) * 1e12)
    fprintf('  Q0.50  : %g\n', q_mc(2) * 1e12)
    fprintf('  Q0.975 : %g\n', q_mc(3) * 1e12)

    figure;
    histogram(Keff_MonteCarlo_m2 * 1e12, 100);
    xlabel('K_{eff} [\mum^2]')
    ylabel('Count')
    title('Monte Carlo distribution of upscaled permeability')
end

%% ------------------------------------------------------------------------
% Save outputs
% -------------------------------------------------------------------------
save_file = ['./outputs/mrst_upscaling_results_' frac_name '.mat'];
save(save_file, ...
    'Keff_MonteCarlo_m2', ...
    'Keff_cubic_law_m2', ...
    'Keff_local_cubic_law_m2', ...
    'Keff_lower_m2', ...
    'Keff_mode_m2', ...
    'Keff_mean_m2', ...
    'Keff_upper_m2')

tEnd = cputime - tStart;
fprintf('\nComputational CPU time [s]: %f\n', tEnd)

%% ========================================================================
% Local functions
% ========================================================================

function Keff = local_computeKeff(G, perm_m2, fluid, mu_visc, bc, facesOut, dpdx, Kmin_m2)
    perm_m2 = perm_m2(:);
    badPerm = ~isfinite(perm_m2) | perm_m2 <= 0;
    perm_m2(badPerm) = Kmin_m2;

    rock = makeRock(G, perm_m2, 1);
    T = computeTrans(G, rock, 'Verbose', false);

    resSol = initResSol(G, 0.0);
    resSol = incompTPFA(resSol, G, T, fluid, 'bc', bc);

    v = sum(resSol.flux(facesOut)) / sum(G.faces.areas(facesOut));
    Keff = -mu_visc * v / dpdx;
end

function [resSol, rock] = solve_darcy(G, perm_m2, fluid, bc, Kmin_m2)
    perm_m2 = perm_m2(:);
    badPerm = ~isfinite(perm_m2) | perm_m2 <= 0;
    perm_m2(badPerm) = Kmin_m2;

    rock = makeRock(G, perm_m2, 1);
    T = computeTrans(G, rock, 'Verbose', false);

    resSol = initResSol(G, 0.0);
    resSol = incompTPFA(resSol, G, T, fluid, 'bc', bc, 'MatrixOutput', true);
end