""" 
constants.py

Universal constants used by files crystals.py, tT_path.py, and tT_model.py 

"""
import numpy as np

#time constants
sec_per_myr = 3.15569259747 * 10**13
sec_per_yr = 3.15569259747 * 10**7

#gas constant in kJ/K * mol-1
gas_constant = 0.008314462618

#jaffey decay constants, 1/s
lambda_232 = np.log(2) / (1.405 * 10**10) * (1 / sec_per_yr)
lambda_235 = np.log(2) / (7.0381 * 10**8) * (1 / sec_per_yr)
lambda_238 = np.log(2) / (4.4683 * 10**9) * (1 / sec_per_yr)

#jaffey decay constants, 1/yr
lambda_232_yr = np.log(2) / (1.405 * 10**10) 
lambda_235_yr = np.log(2) / (7.0381 * 10**8) 
lambda_238_yr = np.log(2) / (4.4683 * 10**9) 

#Sm and FT decay constants, 1/s
lambda_147 = 6.54 * 10**-12 * (1 / sec_per_yr)
lambda_f = 8.46 * 10**-17 * (1 / sec_per_yr)

#Sm and FT decay constants, 1/yr
lambda_147_yr = 6.54 * 10**-12 
lambda_f_yr = 8.46 * 10**-17 

#Avogadro conversion constants
atom_mol = 6.02214076 * 10**23
atom_nmol = 6.02214076 * 10**14

#ppm to atoms/g conversion factor, atomic weights from 2021 IUPAC report, uses 1/137.818 ratio for U isotopes
U235_ppm_atom = (1 / 1000000) * (0.0072559 / 238.02891) * atom_mol
U238_ppm_atom = (1 / 1000000) * (0.99274405 / 238.02891) * atom_mol
Th_ppm_atom = (1 / 1000000) * (1 / 232.0377) * atom_mol
Sm_ppm_atom = (1 / 1000000) * (0.015 / 150.36) * atom_mol

#mineral densities g/cc
ap_density = 3.19
zirc_density = 4.60
