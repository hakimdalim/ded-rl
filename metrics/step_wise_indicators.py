"""
Step-wise metrics for evaluating DED-LB process characteristics.

Each metric function is designed to be applied to individual simulation time steps
or measurements. Functions accept scalar values or arrays and return corresponding
scalar or array outputs.

All metrics include configurable parameters and detailed documentation with formulas.
"""

import numpy as np
from typing import Union, Optional

# Type alias for numeric inputs
Numeric = Union[float, np.ndarray]


def penetration_depth_ratio(
        melt_pool_depth: Numeric,
        track_height: Numeric,
        eps: float = 1e-12
) -> Numeric:
    """
    Calculate penetration depth ratio Ψ_p.

    Quantifies melt pool penetration into previous layer relative to track height.
    Primary estimator of inter-layer lack-of-fusion (LOF) defects.

    Formula:
        Ψ_p = δ_p / h
        where δ_p = d_p - h

    Where:
        δ_p: penetration depth [m]
        d_p: total melt pool depth [m]
        h: actual deposited track height [m]

    Interpretation:
        Ψ_p = 0: No penetration (no bonding)
        0 < Ψ_p << 1: Insufficient bonding, LOF risk
        Ψ_p ≈ 0.2-1.5: Acceptable bonding range (material-dependent)
        Ψ_p >> 1: Excessive remelting, keyhole risk

    Args:
        melt_pool_depth: Total melt pool depth d_p [m]
        track_height: Actual deposited track height h [m]
        eps: Small value to avoid division by zero

    Returns:
        Penetration depth ratio Ψ_p [dimensionless]
        Returns NaN where track_height <= 0

    References:
        - Mukherjee et al. (2016): LOF index definition
        - Carrozza et al. (2021): Growth-to-Depth ratio
    """
    # Calculate penetration depth
    delta_p = melt_pool_depth - track_height

    # Avoid division by zero
    h_safe = np.where(track_height > 0, track_height, np.nan)

    return delta_p / h_safe


def dilution(
        melt_pool_depth: Numeric,
        track_height: Numeric,
        eps: float = 1e-12
) -> Numeric:
    """
    Calculate dilution D.

    Fraction of remelted substrate material within the total fusion zone.
    Geometric approximation based on penetration depth ratio.

    Formula:
        D = δ_p / (h + δ_p) = Ψ_p / (1 + Ψ_p)
        where δ_p = d_p - h, Ψ_p = δ_p / h

    Where:
        δ_p: penetration depth [m]
        d_p: total melt pool depth [m]
        h: actual deposited track height [m]
        Ψ_p: penetration depth ratio [dimensionless]

    Interpretation:
        D < 0.1: Insufficient bonding
        0.1 ≤ D ≤ 0.35: Acceptable range (material-dependent)
        D > 0.35: Excessive mixing, compositional drift

    Args:
        melt_pool_depth: Total melt pool depth d_p [m]
        track_height: Actual deposited track height h [m]
        eps: Small value to avoid division by zero

    Returns:
        Dilution D [dimensionless, 0-1]
        Returns NaN where track_height <= 0

    References:
        - Kaplan & Groboth (2001): Dilution definition for cladding
        - Huang et al. (2019): Geometric approximation for DED
    """
    # First calculate Ψ_p
    psi_p = penetration_depth_ratio(melt_pool_depth, track_height, eps)

    # Calculate dilution from Ψ_p
    return psi_p / (1.0 + psi_p)


def wetting_angle(
        wetting_angle_rad: Numeric
) -> Numeric:
    """
    Convert wetting angle from radians to degrees.

    Contact angle between liquid track and substrate during solidification.
    Governs final track geometry through surface tension effects.

    Formula:
        α_w [°] = α_w [rad] × (180/π)

    Interpretation:
        α_w > 90°: Poor wetting, balling risk
        40° ≤ α_w ≤ 70°: Optimal range for stable tracks
        α_w < 30°: Excessive spreading, edge instability

    Args:
        wetting_angle_rad: Wetting angle in radians [rad]

    Returns:
        Wetting angle in degrees [°]

    References:
        - Huang et al. (2019): Hoffman-Voinov-Tanner law implementation
        - Dos Santos Paes et al. (2021): Wetting behavior in DED
    """
    return np.degrees(wetting_angle_rad)


def track_overlap_ratio(
        track_width: Numeric,
        hatch_spacing: float,
        eps: float = 1e-12
) -> Numeric:
    """
    Calculate track overlap ratio Ψ_o.

    Lateral overlap between adjacent tracks as percentage of track width.
    Critical for ensuring complete surface coverage without excessive heat accumulation.

    Formula:
        Ψ_o = (δ_o / w) × 100%
        where δ_o = w - h_s

    Where:
        w: final track width [m]
        h_s: hatch spacing (center-to-center distance) [m]
        δ_o: overlap distance [m]

    Interpretation:
        Ψ_o < 10%: Insufficient overlap, inter-track voids
        20% ≤ Ψ_o ≤ 40%: Optimal range for full density
        Ψ_o > 50%: Excessive overlap, heat accumulation

    Args:
        track_width: Final track width w [m]
        hatch_spacing: Center-to-center distance between tracks h_s [m]
        eps: Small value to avoid division by zero

    Returns:
        Track overlap ratio Ψ_o [%]
        Returns NaN where track_width <= 0

    References:
        - Established practice in laser cladding and DED
    """
    # Calculate overlap distance
    delta_o = track_width - hatch_spacing

    # Avoid division by zero
    w_safe = np.where(track_width > 0, track_width, np.nan)

    return (delta_o / w_safe) * 100.0


def track_aspect_ratio(
        track_width: Numeric,
        track_height: Numeric,
        eps: float = 1e-12
) -> Numeric:
    """
    Calculate track aspect ratio Ψ_t.

    Geometric proportions of solidified track cross-section.
    Indicates balance between lateral spreading and vertical buildup.

    Formula:
        Ψ_t = w / h

    Where:
        w: final track width [m]
        h: final track height [m]

    Interpretation:
        Ψ_t < 2: Narrow/tall, unstable tracks
        3 ≤ Ψ_t ≤ 7: Optimal range for stable deposition
        Ψ_t > 10: Excessive spreading, low build rate

    Args:
        track_width: Final track width w [m]
        track_height: Final track height h [m]
        eps: Small value to avoid division by zero

    Returns:
        Track aspect ratio Ψ_t [dimensionless]
        Returns NaN where track_height <= 0

    References:
        - Common geometric criterion in DED literature
    """
    # Avoid division by zero
    h_safe = np.where(track_height > 0, track_height, np.nan)

    return track_width / h_safe


def melt_pool_aspect_ratio(
        melt_pool_depth: Numeric,
        melt_pool_width: Numeric,
        eps: float = 1e-12
) -> Numeric:
    """
    Calculate melt pool aspect ratio Ψ_m.

    Shape of molten region distinguishing between melting modes.
    Predicts potential defects from mode transitions.

    Formula:
        Ψ_m = d_p / w_p

    Where:
        d_p: melt pool depth [m]
        w_p: melt pool width at surface [m]

    Interpretation:
        Ψ_m < 0.5: Conduction mode (shallow, stable)
        0.3 ≤ Ψ_m ≤ 0.7: Optimal range for stable processing
        0.5 ≤ Ψ_m < 1.0: Transition mode
        Ψ_m ≥ 1.0: Keyhole mode (deep, unstable, porosity risk)

    Args:
        melt_pool_depth: Melt pool depth d_p [m]
        melt_pool_width: Melt pool width at surface w_p [m]
        eps: Small value to avoid division by zero

    Returns:
        Melt pool aspect ratio Ψ_m [dimensionless]
        Returns NaN where melt_pool_width <= 0

    References:
        - DebRoy et al. (2018): Melting mode transitions in AM
    """
    # Avoid division by zero
    w_p_safe = np.where(melt_pool_width > 0, melt_pool_width, np.nan)

    return melt_pool_depth / w_p_safe


def power_density(
        laser_power: float,
        beam_diameter: float,
        eps: float = 1e-12
) -> float:
    """
    Calculate power density I.

    Instantaneous energy flux delivered to material surface.
    Determines fundamental melting regime independent of scan speed.

    Formula:
        I = P / (π r²) = 4P / (π D_L²)

    Where:
        P: laser power [W]
        D_L: laser beam diameter [m]
        r: beam radius = D_L/2 [m]

    Interpretation:
        I < 10⁵ W/cm²: Conduction mode
        I ≈ 10⁶ W/cm²: Keyhole threshold (material-dependent)
        I > 10⁶ W/cm²: Keyhole mode, vaporization

    Args:
        laser_power: Laser power P [W]
        beam_diameter: Laser beam diameter D_L [m]
        eps: Small value to avoid division by zero

    Returns:
        Power density I [W/m²]

    References:
        - DebRoy et al. (2018): Keyhole formation thresholds
    """
    radius = beam_diameter / 2.0
    area = np.pi * radius ** 2

    # Avoid division by zero
    area_safe = max(area, eps)

    return laser_power / area_safe


def linear_energy_density(
        laser_power: float,
        scan_speed: float,
        eps: float = 1e-12
) -> float:
    """
    Calculate linear energy density E_l.

    Energy input per unit length of scan path.
    Most commonly used process metric correlating with track geometry.

    Formula:
        E_l = P / v

    Where:
        P: laser power [W]
        v: scan speed [m/s]

    Interpretation:
        E_l < 50 J/mm: Incomplete melting, LOF risk
        50 ≤ E_l ≤ 150 J/mm: Typical range for stable tracks
        E_l > 200 J/mm: Excessive energy, keyhole risk

    Args:
        laser_power: Laser power P [W]
        scan_speed: Scan speed v [m/s]
        eps: Small value to avoid division by zero

    Returns:
        Linear energy density E_l [J/m]

    Note:
        To convert to J/mm, divide result by 1000

    References:
        - Fundamental parameter in welding and DED literature
    """
    # Avoid division by zero
    v_safe = max(scan_speed, eps)

    return laser_power / v_safe


def comparison_parameter(
        laser_power: float,
        scan_speed: float,
        powder_feed_rate: float,
        eps: float = 1e-12
) -> float:
    """
    Calculate comparison parameter S.

    Normalizes energy input by powder mass flow rate.
    Accounts for powder heat sink effect, enabling cross-material comparison.

    Formula:
        S = P / (v · ṁ)

    Where:
        P: laser power [W]
        v: scan speed [m/s]
        ṁ: powder mass flow rate [kg/s]

    Interpretation:
        S < 10 J·mm/(g·s): Insufficient energy for full melting
        15 ≤ S ≤ 35 J·mm/(g·s): Optimal range for most materials
        S > 50 J·mm/(g·s): Energy waste, vaporization

    Args:
        laser_power: Laser power P [W]
        scan_speed: Scan speed v [m/s]
        powder_feed_rate: Powder mass flow rate ṁ [kg/s]
        eps: Small value to avoid division by zero

    Returns:
        Comparison parameter S [J·m/(kg·s)] = [W·s·m/kg]

    Note:
        To convert to J·mm/(g·s), multiply result by 1000

    References:
        - Enables process transfer between materials and systems
    """
    # Avoid division by zero
    v_safe = max(scan_speed, eps)
    m_safe = max(powder_feed_rate, eps)

    return laser_power / (v_safe * m_safe)


def interaction_time(
        beam_diameter: float,
        scan_speed: float,
        eps: float = 1e-12
) -> float:
    """
    Calculate interaction time t_i.

    Duration that any point on substrate experiences direct laser irradiation.
    Determines time available for heat conduction and powder melting.

    Formula:
        t_i = D_L / v

    Where:
        D_L: laser beam diameter [m]
        v: scan speed [m/s]

    Interpretation:
        t_i < 0.01 s: May cause incomplete powder melting
        0.1 ≤ t_i ≤ 0.5 s: Typical range for complete melting
        t_i > 0.5 s: Risk of excessive heat accumulation

    Args:
        beam_diameter: Laser beam diameter D_L [m]
        scan_speed: Scan speed v [m/s]
        eps: Small value to avoid division by zero

    Returns:
        Interaction time t_i [s]

    References:
        - Related to Peclet number Pe = vD_L/(2α) for heat transfer regime
    """
    # Avoid division by zero
    v_safe = max(scan_speed, eps)

    return beam_diameter / v_safe


def specific_energy(
        laser_power: float,
        powder_feed_rate: float,
        eps: float = 1e-12
) -> float:
    """
    Calculate specific energy E_s.

    Energy available per unit mass of powder feedstock.
    Indicates whether sufficient energy exists for melting, independent of scan speed.

    Formula:
        E_s = P / ṁ

    Where:
        P: laser power [W]
        ṁ: powder mass flow rate [kg/s]

    Minimum theoretical requirement:
        E_s,min = c_p(T_m - T_0) + L_f

    Where:
        c_p: specific heat capacity [J/(kg·K)]
        T_m: melting temperature [K]
        T_0: initial powder temperature [K]
        L_f: latent heat of fusion [J/kg]

    Interpretation (for steel):
        E_s < 1500 J/g: Insufficient for complete melting
        1500 ≤ E_s ≤ 3000 J/g: Adequate energy with margin
        E_s > 3000 J/g: Risk of vaporization

    Args:
        laser_power: Laser power P [W]
        powder_feed_rate: Powder mass flow rate ṁ [kg/s]
        eps: Small value to avoid division by zero

    Returns:
        Specific energy E_s [J/kg]

    Note:
        To convert to J/g, divide result by 1000

    References:
        - Fundamental thermodynamic requirement for melting
    """
    # Avoid division by zero
    m_safe = max(powder_feed_rate, eps)

    return laser_power / m_safe


def cooling_rate_from_solidification(
        temperature_gradient: Numeric,
        solidification_rate: Numeric,
        eps: float = 1e-12
) -> Numeric:
    """
    Calculate cooling rate from solidification parameters (G × R).

    Product of temperature gradient and solidification rate.
    Controls microstructure scale through dendrite arm spacing.

    Formula:
        dT/dt = G × R

    Where:
        G: temperature gradient at solidification front [K/m]
        R: solidification rate [m/s]

    Relationships:
        Primary dendrite arm spacing: λ₁ ∝ (G×R)^(-0.25)
        Secondary dendrite arm spacing: λ₂ ∝ (G×R)^(-0.33)

    Interpretation:
        G×R < 10² K/s: Coarse microstructure, reduced strength
        10² ≤ G×R ≤ 10⁴ K/s: Typical DED range
        G×R > 10⁴ K/s: Fine microstructure, high residual stress

    Args:
        temperature_gradient: Temperature gradient G [K/m]
        solidification_rate: Solidification rate R [m/s]
        eps: Small value to avoid numerical issues

    Returns:
        Cooling rate dT/dt [K/s]

    References:
        - Kurz & Fisher (1998): Fundamentals of Solidification
        - DebRoy et al. (2018): AM microstructure relationships
    """
    return temperature_gradient * solidification_rate


def morphology_ratio(
        temperature_gradient: Numeric,
        solidification_rate: Numeric,
        eps: float = 1e-12
) -> Numeric:
    """
    Calculate morphology ratio (G/R).

    Determines solidification mode through constitutional supercooling effects.
    Controls transition between columnar and equiaxed grain structures.

    Formula:
        G/R

    Where:
        G: temperature gradient at solidification front [K/m]
        R: solidification rate [m/s]

    Constitutional supercooling criterion:
        G/R > (m·C₀·(1-k₀))/(D_l·k₀)

    Where:
        m: liquidus slope [K/wt%]
        C₀: alloy composition [wt%]
        k₀: partition coefficient [dimensionless]
        D_l: liquid diffusivity [m²/s]

    Interpretation:
        G/R > 10⁴ K·s/m²: Planar solidification (rare in DED)
        10² ≤ G/R ≤ 10⁴ K·s/m²: Cellular/columnar dendritic
        G/R < 10² K·s/m²: Equiaxed dendritic

    Args:
        temperature_gradient: Temperature gradient G [K/m]
        solidification_rate: Solidification rate R [m/s]
        eps: Small value to avoid division by zero

    Returns:
        Morphology ratio G/R [K·s/m²]

    References:
        - Kurz & Fisher (1998): Constitutional supercooling criterion
        - Hunt (1984): Columnar-to-equiaxed transition
    """
    # Avoid division by zero
    R_safe = np.where(solidification_rate > eps, solidification_rate, np.nan)

    return temperature_gradient / R_safe


def constitutional_supercooling_criterion(
        temperature_gradient: Numeric,
        solidification_rate: Numeric,
        solidification_range: float,
        liquid_diffusivity: float,
        eps: float = 1e-12
) -> Numeric:
    """
    Calculate constitutional supercooling criterion Ψ_cs.

    Evaluates whether liquid ahead of solidification front becomes supercooled.
    Predicts destabilization of planar solidification interface.

    Formula:
        Ψ_cs = (G/R) / (ΔT_l/D_l)

    Where:
        G: temperature gradient [K/m]
        R: solidification rate [m/s]
        ΔT_l: solidification temperature range T_l - T_s [K]
        D_l: solute diffusivity in liquid [m²/s]

    Interpretation:
        Ψ_cs > 1: No constitutional supercooling, planar front stable
        Ψ_cs < 1: Constitutional supercooling occurs, cellular/dendritic growth

    Typical DED range: 0.01 < Ψ_cs < 0.5 (dendritic solidification prevalent)

    Args:
        temperature_gradient: Temperature gradient G [K/m]
        solidification_rate: Solidification rate R [m/s]
        solidification_range: Temperature range ΔT_l = T_liquidus - T_solidus [K]
        liquid_diffusivity: Solute diffusivity D_l [m²/s]
        eps: Small value to avoid division by zero

    Returns:
        Constitutional supercooling criterion Ψ_cs [dimensionless]

    References:
        - Tiller et al. (1953): Original constitutional supercooling theory
        - Kurz & Fisher (1998): Solidification morphology prediction
    """
    # Calculate G/R
    G_over_R = morphology_ratio(temperature_gradient, solidification_rate, eps)

    # Calculate threshold ΔT_l/D_l
    threshold = solidification_range / max(liquid_diffusivity, eps)

    return G_over_R / threshold


def mushy_zone_size(
        solidification_range: float,
        temperature_gradient: Numeric,
        eps: float = 1e-12
) -> Numeric:
    """
    Calculate mushy zone size δ_m.

    Width of partially solidified region between liquidus and solidus isotherms.
    Directly indicates hot cracking susceptibility.

    Formula (centerline approximation):
        δ_m = ΔT_l / G

    Where:
        ΔT_l: solidification temperature range T_l - T_s [K]
        G: temperature gradient [K/m]

    Interpretation:
        δ_m < 0.1 mm: Narrow, may cause shrinkage porosity
        0.1 ≤ δ_m ≤ 0.5 mm: Optimal range balancing feeding and strength
        δ_m > 1 mm: Wide, increased hot cracking susceptibility

    Args:
        solidification_range: Temperature range ΔT_l = T_liquidus - T_solidus [K]
        temperature_gradient: Temperature gradient G [K/m]
        eps: Small value to avoid division by zero

    Returns:
        Mushy zone size δ_m [m]

    Note:
        Multiply by 1000 to convert to mm

    References:
        - Rappaz et al. (1999): Hot cracking mechanisms
        - Kou (2003): Welding metallurgy susceptibility criteria
    """
    # Avoid division by zero
    G_safe = np.where(temperature_gradient > eps, temperature_gradient, np.nan)

    return solidification_range / G_safe


def peak_temperature_compliance(
        max_temperature: Numeric,
        melting_temperature: float,
        boiling_temperature: float
) -> Numeric:
    """
    Check if peak temperature is within acceptable range.

    Fundamental constraint for successful deposition.
    Peak temperature must exceed melting point for fusion but remain
    below boiling point to avoid vaporization and keyholing.

    Formula:
        Compliant if: T_m < T_max < T_b

    Where:
        T_max: maximum temperature in melt pool [K]
        T_m: melting temperature [K]
        T_b: boiling temperature [K]

    Interpretation:
        T_max < T_m: No melt pool formation (insufficient energy)
        T_m < T_max < T_b: Acceptable range for stable processing
        T_max ≥ T_b: Vaporization, keyhole porosity risk

    Typical operating range: 1.1*T_m < T_max < 0.9*T_b

    Args:
        max_temperature: Maximum temperature reached T_max [K]
        melting_temperature: Material melting temperature T_m [K]
        boiling_temperature: Material boiling temperature T_b [K]

    Returns:
        Boolean array: True where temperature is in acceptable range
        For scalar inputs, returns single boolean

    References:
        - Fundamental thermodynamic constraint for melting
        - DebRoy et al. (2018): Keyhole formation thresholds
    """
    above_melting = max_temperature > melting_temperature
    below_boiling = max_temperature < boiling_temperature

    return np.logical_and(above_melting, below_boiling)