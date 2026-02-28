import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime, timezone,timedelta

import cartopy.crs as ccrs
import cartopy.feature as cfeature


# ================================================================
# Constants
# ================================================================
mu = 398600.4418          # km^3/s^2
omega_earth = 7.2921159e-5
Re = 6378.137             # km
c = 299792.458            # km/s
epoch0=datetime(2000,1,1,21,0,0,tzinfo=timezone.utc)

epoch_current=datetime(2026,2,27,12,0,0, tzinfo=timezone.utc)+ timedelta(seconds=18527)

v1state=(epoch_current,7000,0.71,30,40,0,(45,-90),1e9,9.0)
v2state=(epoch_current,7600,0.08,35,60,10,(45,-90),1e9,2.4)

# ================================================================
# Satellite state
# ================================================================

from scipy.optimize import newton
import numpy as np

def build_vehicle_state(epoch,r_perigee, e, i_deg, RAAN_deg, arg_perigee_deg, target_latlon, tuner_freq, dish_size):
    # --- Orbital parameters ---
    i = np.deg2rad(i_deg)
    RAAN = np.deg2rad(RAAN_deg)
    arg_perigee = np.deg2rad(arg_perigee_deg)
    a = r_perigee / (1 - e)          # semi-major axis

    # --- Mean motion ---
    n = np.sqrt(mu / a**3)  # rad/s

    # --- Time since perigee (assume perigee at epoch0) ---
    t_since_perigee = (epoch - epoch0).total_seconds()

    # --- Mean anomaly ---
    M = n * t_since_perigee

    # --- Solve Kepler's equation: M = E - e*sin(E) ---
    E = newton(lambda E: E - e*np.sin(E) - M, M)

    # --- True anomaly ---
    nu = 2*np.arctan2(np.sqrt(1+e)*np.sin(E/2), np.sqrt(1-e)*np.cos(E/2))

    # --- Radius at true anomaly ---
    r_mag = a * (1 - e*np.cos(E))

    # --- Perifocal position & velocity ---
    r_pf = np.array([r_mag*np.cos(nu), r_mag*np.sin(nu), 0])
    v_pf = np.array([-np.sin(E), np.sqrt(1-e**2)*np.cos(E), 0]) * np.sqrt(mu*a)/r_mag

    # --- Rotation to ECI ---
    R3_W = np.array([[np.cos(RAAN), -np.sin(RAAN), 0],
                     [np.sin(RAAN), np.cos(RAAN), 0],
                     [0, 0, 1]])
    R1_i = np.array([[1,0,0],[0,np.cos(i),-np.sin(i)],[0,np.sin(i),np.cos(i)]])
    R3_w = np.array([[np.cos(arg_perigee), -np.sin(arg_perigee), 0],
                     [np.sin(arg_perigee), np.cos(arg_perigee), 0],
                     [0,0,1]])
    Q = R3_W @ R1_i @ R3_w
    r_eci = Q @ r_pf
    v_eci = Q @ v_pf

    # --- Rotate to ECR using GMST ---
    JD = 2451545.0 + (epoch - epoch0).total_seconds()/86400
    GMST = np.deg2rad((280.46061837 + 360.98564736629*(JD-2451545.0)) % 360)
    R3_gmst = np.array([[ np.cos(GMST), np.sin(GMST),0],
                        [-np.sin(GMST), np.cos(GMST),0],
                        [0,0,1]])
    r_ecr = R3_gmst @ r_eci
    v_ecr = R3_gmst @ (v_eci - np.cross([0,0,omega_earth], r_eci))

    return {
        "epoch": epoch,
        "r_ecr": r_ecr,
        "v_ecr": v_ecr,
        "target_location": target_latlon,
        "tuner_center_frequency": tuner_freq,
        "dish_diameter": dish_size
    }


# ================================================================
# Orbit path computation
# ================================================================
def compute_orbit_path_3d(epoch,r_perigee, e, i_deg, RAAN_deg, arg_perigee_deg, target,Fc,ant_dia,num_points=360):
    nu = np.linspace(0,2*np.pi,num_points)
    a = r_perigee / (1 - e)
    r = a*(1-e**2)/(1+e*np.cos(nu))
    r_pf = np.vstack([r*np.cos(nu), r*np.sin(nu), np.zeros_like(nu)])
    i = np.deg2rad(i_deg)
    RAAN = np.deg2rad(RAAN_deg)
    arg_perigee = np.deg2rad(arg_perigee_deg)
    R3_W = np.array([[np.cos(RAAN),-np.sin(RAAN),0],[np.sin(RAAN),np.cos(RAAN),0],[0,0,1]])
    R1_i = np.array([[1,0,0],[0,np.cos(i),-np.sin(i)],[0,np.sin(i),np.cos(i)]])
    R3_w = np.array([[np.cos(arg_perigee),-np.sin(arg_perigee),0],[np.sin(arg_perigee),np.cos(arg_perigee),0],[0,0,1]])
    Q = R3_W @ R1_i @ R3_w
    r_eci = Q @ r_pf

    JD = 2451545.0 + (epoch - epoch0).total_seconds()/86400
    GMST = np.deg2rad((280.46061837 + 360.98564736629*(JD-2451545.0)) % 360)
    R3_gmst = np.array([[ np.cos(GMST), np.sin(GMST),0],
                        [-np.sin(GMST), np.cos(GMST),0],
                        [0,0,1]])
    return R3_gmst @ r_eci

# ================================================================
# ECR to lat/lon
# ================================================================
def ecr_to_latlon(r_ecr):
    x,y,z = r_ecr
    lat = np.arcsin(z/np.linalg.norm(r_ecr))
    lon = np.arctan2(y,x)
    return np.rad2deg(lat), np.rad2deg(lon)

# ================================================================
# Ground track
# ================================================================
def compute_ground_track(epoch,r_perigee,e,i_deg,RAAN_deg,arg_perigee_deg,loc,Fc,ant_dia,num_points=360):
    r = compute_orbit_path_3d(epoch,r_perigee,e,i_deg,RAAN_deg,arg_perigee_deg,loc,Fc,ant_dia,num_points)
    lat = np.arcsin(r[2,:]/np.linalg.norm(r,axis=0))
    lon = np.arctan2(r[1,:],r[0,:])
    return np.rad2deg(lat), np.rad2deg(lon)

# ================================================================
#  FOV cone oriented toward target
# ================================================================
def fov_circle_target(vehicle, num_points=100):
    c_m_s = 299792458.0
    r_sat = vehicle['r_ecr']
    target_lat, target_lon = vehicle['target_location']
    lat_rad = np.deg2rad(target_lat)
    lon_rad = np.deg2rad(target_lon)
    r_target = np.array([Re*np.cos(lat_rad)*np.cos(lon_rad),
                         Re*np.cos(lat_rad)*np.sin(lon_rad),
                         Re*np.sin(lat_rad)])
    
    v_axis = r_target - r_sat
    v_axis /= np.linalg.norm(v_axis)
    
    wavelength = c_m_s / vehicle['tuner_center_frequency']
    theta_beam = 1.22 * wavelength / vehicle['dish_diameter']
    
    h = np.linalg.norm(r_sat) - Re
    
    angles = np.linspace(0,2*np.pi,num_points)
    circle_points = np.zeros((num_points,3))
    
    if np.allclose(v_axis,[0,0,1]):
        u = np.array([1,0,0])
    else:
        u = np.cross([0,0,1],v_axis)
        u /= np.linalg.norm(u)
    v = np.cross(v_axis,u)
    
    for i, alpha in enumerate(angles):
        dir_vec = np.cos(alpha)*u + np.sin(alpha)*v
        dir_vec = np.cos(theta_beam)*v_axis + np.sin(theta_beam)*dir_vec
        dir_vec /= np.linalg.norm(dir_vec)
        a = np.dot(dir_vec, dir_vec)
        b = 2*np.dot(dir_vec, r_sat)
        c_eq = np.dot(r_sat,r_sat)-Re**2
        disc = b**2 -4*a*c_eq
        if disc < 0:
            circle_points[i,:] = np.nan
        else:
            t = (-b - np.sqrt(disc)) / (2*a)
            circle_points[i,:] = r_sat + t*dir_vec
    
    lat = np.arcsin(circle_points[:,2]/Re)
    lon = np.arctan2(circle_points[:,1],circle_points[:,0])
    return np.rad2deg(lat), np.rad2deg(lon)

# ================================================================
# Create vehicles and orbits
# ================================================================

vehicle1 = build_vehicle_state(*v1state)
vehicle2 = build_vehicle_state(*v2state)

# Compute nadirs
nadir1 = np.array(ecr_to_latlon(vehicle1['r_ecr']))
nadir2 = np.array(ecr_to_latlon(vehicle2['r_ecr']))
common_target = ((nadir1 + nadir2)/2).tolist()  # midpoint for visibility

# Rebuild vehicles with updated target
v1temp=[tt for tt in v1state]
v1temp[6]=common_target
vehicle1 = build_vehicle_state(*v1temp)

v2temp=[tt for tt in v2state]
v2temp[6]=common_target
vehicle2 = build_vehicle_state(*v2temp)



orbit1_path = compute_orbit_path_3d(*v1state)
orbit2_path = compute_orbit_path_3d(*v2state)

# ================================================================
# 3D orbit plot
# ================================================================
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111,projection='3d')

# Earth
theta = np.linspace(0,2*np.pi,30)
phi = np.linspace(0,np.pi,20)
THETA,PHI = np.meshgrid(theta,phi)
X = Re*np.sin(PHI)*np.cos(THETA)
Y = Re*np.sin(PHI)*np.sin(THETA)
Z = Re*np.cos(PHI)
ax.plot_surface(X,Y,Z,color='green',alpha=1)

# Orbits
ax.plot(orbit1_path[0,:],orbit1_path[1,:],orbit1_path[2,:],color='blue',label='Vehicle 1 Orbit')
ax.plot(orbit2_path[0,:],orbit2_path[1,:],orbit2_path[2,:],color='red',label='Vehicle 2 Orbit')

# Current positions
ax.scatter(*vehicle1['r_ecr'],color='blue',s=100,label='Vehicle 1 Now')
ax.scatter(*vehicle2['r_ecr'],color='red',s=100,label='Vehicle 2 Now')

ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.set_title('3D Orbits')
ax.legend()
ax.set_box_aspect([1,1,1])

# ================================================================
#  TDOA, TDOA-dot, TDOA-dot-dot
# ================================================================

fig, axes = plt.subplots(1, 3, figsize=(24, 6), constrained_layout=True,
                         subplot_kw={'projection': ccrs.PlateCarree()})


lat_grid = np.linspace(-60,60,121)
lon_grid = np.linspace(-180,180,361)
LAT,LON = np.meshgrid(lat_grid,lon_grid)
LAT_rad = np.deg2rad(LAT)
LON_rad = np.deg2rad(LON)

r_earth_grid = np.zeros(LAT.shape + (3,))
r_earth_grid[...,0] = Re*np.cos(LAT_rad)*np.cos(LON_rad)
r_earth_grid[...,1] = Re*np.cos(LAT_rad)*np.sin(LON_rad)
r_earth_grid[...,2] = Re*np.sin(LAT_rad)

# Nadir points
nadir1 = ecr_to_latlon(vehicle1['r_ecr'])
nadir2 = ecr_to_latlon(vehicle2['r_ecr'])
plt.scatter(nadir1[1],nadir1[0],color='blue',s=100,marker='o',label='Vehicle 1 Nadir')
plt.scatter(nadir2[1],nadir2[0],color='red',s=100,marker='o',label='Vehicle 2 Nadir')

# Ground tracks
gt_lat1,gt_lon1 = compute_ground_track(*v1state)
gt_lat2,gt_lon2 = compute_ground_track(*v2state)

lat_fov1, lon_fov1 = fov_circle_target(vehicle1)
lat_fov2, lon_fov2 = fov_circle_target(vehicle2)
mask1 = ~np.isnan(lat_fov1)
mask2 = ~np.isnan(lat_fov2)


# Flatten grids for vectorized computations
r_flat = r_earth_grid.reshape(-1,3)
LAT_flat = LAT.ravel()
LON_flat = LON.ravel()

# Distances and vectors
vec1 = r_flat - vehicle1['r_ecr']
vec2 = r_flat - vehicle2['r_ecr']
dist1 = np.linalg.norm(vec1, axis=1)
dist2 = np.linalg.norm(vec2, axis=1)

# --- TDOA ---
TDOA_flat = (dist1 - dist2)/c*1e3
sc0 = axes[0].scatter(LON_flat, LAT_flat, c=TDOA_flat, cmap='RdBu', s=10, edgecolors='none', transform=ccrs.PlateCarree())
axes[0].set_title('TDOA (ms)')
axes[0].add_feature(cfeature.LAND, facecolor='lightgray')
axes[0].add_feature(cfeature.OCEAN, facecolor='lightblue')
axes[0].add_feature(cfeature.COASTLINE)
axes[0].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
plt.colorbar(sc0, ax=axes[0], label='TDOA (ms)')


# --- TDOA-dot ---
TDOAdot_flat = (np.einsum('ij,j->i', vec1, vehicle1['v_ecr'])/dist1 -
                np.einsum('ij,j->i', vec2, vehicle2['v_ecr'])/dist2) / c*1e6
sc1 = axes[1].scatter(LON_flat, LAT_flat, c=TDOAdot_flat, cmap='RdBu', s=10, edgecolors='none', transform=ccrs.PlateCarree())
axes[1].set_title('TDOA-dot (us/s)')
axes[1].add_feature(cfeature.LAND, facecolor='lightgray')
axes[1].add_feature(cfeature.OCEAN, facecolor='lightblue')
axes[1].add_feature(cfeature.COASTLINE)
axes[1].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
plt.colorbar(sc1, ax=axes[1], label='TDOA-dot (us/s)')

# --- TDOA-dot-dot ---
a1 = -mu * vehicle1['r_ecr'] / np.linalg.norm(vehicle1['r_ecr'])**3
a2 = -mu * vehicle2['r_ecr'] / np.linalg.norm(vehicle2['r_ecr'])**3
TDOAdd_flat = (
    (np.einsum('ij,j->i', vec1, a1) + np.sum(vehicle1['v_ecr']**2)) / dist1
    - (np.einsum('ij,j->i', vec2, a2) + np.sum(vehicle2['v_ecr']**2)) / dist2
    - (np.einsum('ij,j->i', vec1, vehicle1['v_ecr'])**2) / dist1**3
    + (np.einsum('ij,j->i', vec2, vehicle2['v_ecr'])**2) / dist2**3
) / c *1e9
sc2 = axes[2].scatter(LON_flat, LAT_flat, c=TDOAdd_flat, cmap='RdBu', s=10, edgecolors='none', transform=ccrs.PlateCarree())
axes[2].set_title('TDOA-dot-dot (ns/s²)')
axes[2].add_feature(cfeature.LAND, facecolor='lightgray')
axes[2].add_feature(cfeature.OCEAN, facecolor='lightblue')
axes[2].add_feature(cfeature.COASTLINE)
axes[2].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
plt.colorbar(sc2, ax=axes[2], label='TDOA-dot-dot (ns/s²)')

# --- Overlays for all subplots ---
for ax in axes:
    # Nadirs
    ax.scatter(nadir1[1], nadir1[0], color='blue', s=100, marker='o', label='Vehicle 1 Nadir', transform=ccrs.PlateCarree())
    ax.scatter(nadir2[1], nadir2[0], color='red', s=100, marker='o', label='Vehicle 2 Nadir', transform=ccrs.PlateCarree())
    # Ground tracks (geodesic for proper curvature)
    ax.plot(gt_lon1, gt_lat1, color='blue', linestyle='--', alpha=0.7, label='Vehicle 1 Ground Track', transform=ccrs.Geodetic())
    ax.plot(gt_lon2, gt_lat2, color='red', linestyle='--', alpha=0.7, label='Vehicle 2 Ground Track', transform=ccrs.Geodetic())
    # Common target
    ax.scatter(common_target[1], common_target[0], color='gold', s=100, marker='*', label='Common Target', transform=ccrs.PlateCarree())
    # FOV outlines
    ax.plot(lon_fov1[mask1], lat_fov1[mask1], color='blue', linewidth=1.5, label='Vehicle 1 FOV', transform=ccrs.PlateCarree())
    ax.plot(lon_fov2[mask2], lat_fov2[mask2], color='red', linewidth=1.5, label='Vehicle 2 FOV', transform=ccrs.PlateCarree())
    ax.set_extent([-180,180,-90,90], crs=ccrs.PlateCarree())
    #ax.legend(fontsize=8)
    
plt.show()

