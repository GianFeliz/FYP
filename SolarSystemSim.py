import pygame
import numpy as np

# --- Constants ---
G = 6.67430e-11
WIDTH, HEIGHT = 650, 650
BASE_DT = 1200  
AU = 1.496e11   # Astronomical Unit in meters
DAY_SECONDS = 86400
YEAR_SECONDS = 365.25 * DAY_SECONDS
MU_SUN = G * 1.989e30

# --- Asteroid toggle ---
asteroids_on = False

# --- Body class ---
class Body:
    def __init__(self, name, mass, pos, vel, color, radius):
        self.name = name
        self.mass = mass
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.color = color
        self.radius = radius
        self.traj = [self.pos.copy()]
        self.earth_relative_traj = []
        self.timestep_group = 0
        self.dt = BASE_DT
        self.next_update_time = 0.0
        self.half_vel = np.array(vel, dtype=float)
        self.a = 0
        self.e = 0

class Spacecraft(Body):
    def __init__(self, name, mass, pos, vel, color, radius):
        super().__init__(name, mass, pos, vel, color, radius)
        self.burns_done = 0
        self.target_planet = None
        self.has_circularized = False
        # Bi-elliptic fields
        self.rb = 0.0
        self.rb_au = 0.0
        self.transfer_type = 'H'   # 'H' = Hohmann, 'B' = Bi-elliptic
        self.transfer_data = {}

# --- Timestep group assignment ---
def assign_timestep_group(body, sun_pos, sun_mass):
    if body.name in ["Sun", "Ares"] or "Ares" in body.name:
        return 1
    r = np.linalg.norm(body.pos - sun_pos)
    if r == 0:
        return 1
    period_seconds = 2 * np.pi * np.sqrt(r**3 / (G * sun_mass))
    period_days = period_seconds / DAY_SECONDS
    if period_days < 30:
        return 1
    elif period_days < 365:
        return 2
    elif period_days < 730:
        return 3
    elif period_days < 365 * 12:
        return 4
    elif period_days < 365 * 100:
        return 5
    else:
        return 6

GROUP_MULTIPLIERS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}

# --- Rotation matrix ---
def rotation_matrix_x(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

# --- Initialize Sun and planets ---
sun = Body("Sun", 1.989e30, [0,0,0], [0,0,0], (255,255,0), 20)

planet_params = [
    ("Mercury", 3.285e23, 5.791e10, 0.2056, 7.0,  (200,200,200), 3),
    ("Venus",   4.867e24, 1.082e11, 0.0068, 3.4,  (0,255,0),     5),
    ("Earth",   5.972e24, 1.496e11, 0.0167, 0.0,  (0,0,255),     5),
    ("Mars",    6.39e23,  2.279e11, 0.0934, 1.9,  (255,0,0),     4),
    ("Jupiter", 1.898e27, 7.786e11, 0.0489, 1.3,  (255,165,0),   15),
    ("Saturn",  5.683e26, 1.432e12, 0.0565, 2.5,  (210,180,140), 12),
    ("Uranus",  8.682e25, 2.871e12, 0.0472, 0.8,  (173,216,230), 9),
    ("Neptune", 1.024e26, 4.495e12, 0.0086, 1.8,  (70,130,180),  9)
]

bodies = [sun]
planet_list = []

for name, mass, a, e, incl_deg, color, radius in planet_params:
    r_peri = a * (1 - e)
    v_peri = np.sqrt(G * sun.mass * (2 / r_peri - 1 / a))
    inclination = np.radians(incl_deg)
    pos = np.array([r_peri, 0, 0])
    vel = np.array([0, v_peri, 0])
    rot = rotation_matrix_x(inclination)
    pos3d = rot @ pos
    vel3d = rot @ vel
    body = Body(name, mass, pos3d, vel3d, color, radius)
    body.a = a
    body.e = e
    bodies.append(body)
    planet_list.append(body)

earth   = next(b for b in bodies if b.name == "Earth")
mercury = next(b for b in bodies if b.name == "Mercury")
venus   = next(b for b in bodies if b.name == "Venus")
mars    = next(b for b in bodies if b.name == "Mars")
jupiter = next(b for b in bodies if b.name == "Jupiter")
saturn  = next(b for b in bodies if b.name == "Saturn")
uranus  = next(b for b in bodies if b.name == "Uranus")
neptune = next(b for b in bodies if b.name == "Neptune")

TARGET_MAP = {
    pygame.K_1: mercury,
    pygame.K_2: venus,
    pygame.K_3: mars,
    pygame.K_4: jupiter,
    pygame.K_5: saturn,
    pygame.K_6: uranus,
    pygame.K_7: neptune,
}

# --- Initialize Moon ---
moon_dist  = 3.84e8
moon_mass  = 7.342e22
moon_speed = np.sqrt(G * earth.mass / moon_dist)
moon_pos   = earth.pos + np.array([moon_dist, 0, 0])
moon_vel   = earth.vel + np.array([0, moon_speed, 0])
moon = Body("Moon", moon_mass, moon_pos, moon_vel, (200,200,200), 2)
moon.earth_relative_traj = [moon.pos - earth.pos]
bodies.append(moon)

# --- Asteroids ---
asteroid_count = 750
asteroids = []

def generate_asteroids(n):
    global asteroids
    asteroids = []
    for _ in range(n):
        a   = np.random.uniform(2.2, 3.2) * AU
        e   = np.random.uniform(0, 0.15)
        r   = a * (1 - e)
        ang = np.random.uniform(0, 2*np.pi)
        pos = np.array([r*np.cos(ang), r*np.sin(ang), 0])
        v   = np.sqrt(G * sun.mass * (2/r - 1/a))
        vel = np.array([-v*np.sin(ang), v*np.cos(ang), 0])
        asteroids.append(Body("Ast", 1e15, pos, vel, (150,150,150), np.random.uniform(1,2)))

if asteroids_on:
    generate_asteroids(asteroid_count)

all_bodies = bodies + asteroids

# --- Spacecraft (created on demand) ---
spacecraft = None

# --- Acceleration ---
def compute_acceleration_for_body(body, bodies_list):
    total_force = np.zeros(3)
    for other in bodies_list:
        if body is other:
            continue
        if body.mass < 1e20 and other.mass < 1e20:
            continue
        r_vec = other.pos - body.pos
        r_mag = np.linalg.norm(r_vec)
        if r_mag == 0:
            continue
        total_force += G * body.mass * other.mass * r_vec / r_mag**3
    return total_force / body.mass if body.mass > 0 else np.zeros(3)

# --- Timestep groups ---
def initialize_timestep_groups():
    for body in all_bodies:
        body.timestep_group = assign_timestep_group(body, sun.pos, sun.mass)
        body.dt = BASE_DT * GROUP_MULTIPLIERS[body.timestep_group]
        body.next_update_time = 0.0
        acc = compute_acceleration_for_body(body, all_bodies)
        body.half_vel = body.vel + 0.5 * acc * body.dt

initialize_timestep_groups()

# --- Block timestep update ---
def update_bodies(current_time, timewarp=1):
    for _ in range(timewarp):
        for body in all_bodies:
            if body.timestep_group == 0:
                continue
            if timewarp > 1:
                body.pos += body.half_vel * BASE_DT
                acc = compute_acceleration_for_body(body, all_bodies)
                body.half_vel += acc * BASE_DT
                body.vel = body.half_vel - 0.5 * acc * BASE_DT
                body.next_update_time += BASE_DT
            else:
                if current_time >= body.next_update_time:
                    body.pos += body.half_vel * body.dt
                    acc = compute_acceleration_for_body(body, all_bodies)
                    body.half_vel += acc * body.dt
                    body.vel = body.half_vel - 0.5 * acc * body.dt
                    body.next_update_time += body.dt
            body.traj.append(body.pos.copy())
            if len(body.traj) > 300:
                body.traj = body.traj[-300:]
    moon.earth_relative_traj.append((moon.pos - earth.pos).copy())
    if len(moon.earth_relative_traj) > 2000:
        moon.earth_relative_traj = moon.earth_relative_traj[-2000:]

# --- Sun-only acceleration ---
def compute_acc_sun_only(pos):
    r_vec = sun.pos - pos
    r_mag = np.linalg.norm(r_vec)
    if r_mag == 0:
        return np.zeros(3)
    return G * sun.mass * r_vec / r_mag**3

# --- Spacecraft update (leapfrog, sun-only gravity) ---
def update_spacecraft(sc, timewarp=1):
    for _ in range(timewarp):
        sc.pos += sc.half_vel * BASE_DT
        acc = compute_acc_sun_only(sc.pos)
        sc.half_vel += acc * BASE_DT
        sc.vel = sc.half_vel - 0.5 * acc * BASE_DT
        sc.traj.append(sc.pos.copy())
        if len(sc.traj) > 2000:
            sc.traj = sc.traj[-2000:]

# =============================================================================
# --- Hohmann transfer calculations ---
# =============================================================================
def calculate_hohmann_transfer(r1, r2, mu):
    a_transfer = (r1 + r2) / 2
    v_c1 = np.sqrt(mu / r1)
    v_c2 = np.sqrt(mu / r2)
    v_t1 = np.sqrt(mu * (2/r1 - 1/a_transfer))
    v_t2 = np.sqrt(mu * (2/r2 - 1/a_transfer))
    dv1 = abs(v_t1 - v_c1)
    dv2 = abs(v_t2 - v_c2)
    t_transfer = np.pi * np.sqrt(a_transfer**3 / mu)
    return a_transfer, v_t1, v_t2, dv1, dv2, t_transfer

def calculate_phase_angle_for_transfer(r1, r2, mu):
    a_transfer = (r1 + r2) / 2
    t_transfer = np.pi * np.sqrt(a_transfer**3 / mu)
    omega_target = np.sqrt(mu / r2**3)
    phase_angle_rad = np.pi - omega_target * t_transfer
    return np.degrees(phase_angle_rad) % 360

def get_phase_angle(body1, body2):
    angle1 = np.arctan2(body1.pos[1], body1.pos[0])
    angle2 = np.arctan2(body2.pos[1], body2.pos[0])
    diff = (angle2 - angle1) % (2 * np.pi)
    return np.degrees(diff)

# =============================================================================
# --- Bi-elliptic transfer calculations ---
# =============================================================================
def calculate_bi_elliptic(r1, r2, rb):
    """Calculate all delta-vs and timings for a bi-elliptic transfer."""
    a1 = (r1 + rb) / 2
    v_c1 = np.sqrt(MU_SUN / r1)
    v_at_r1_on_a1 = np.sqrt(MU_SUN * (2/r1 - 1/a1))
    dv1 = abs(v_at_r1_on_a1 - v_c1)

    v_at_rb_on_a1 = np.sqrt(MU_SUN * (2/rb - 1/a1))

    a2 = (rb + r2) / 2
    v_at_rb_on_a2 = np.sqrt(MU_SUN * (2/rb - 1/a2))
    dv2 = abs(v_at_rb_on_a2 - v_at_rb_on_a1)

    v_c2 = np.sqrt(MU_SUN / r2)
    v_at_r2_on_a2 = np.sqrt(MU_SUN * (2/r2 - 1/a2))
    dv3 = abs(v_c2 - v_at_r2_on_a2)

    t1 = np.pi * np.sqrt(a1**3 / MU_SUN)
    t2 = np.pi * np.sqrt(a2**3 / MU_SUN)
    t_total = t1 + t2

    return dv1, dv2, dv3, t_total, a1, a2, v_at_rb_on_a1, v_at_rb_on_a2, v_at_r2_on_a2, v_c1, v_c2

def get_bi_elliptic_phase_info(r1, r2, rb, current_time=0):
    """
    Compute the phase error for the bi-elliptic window at the current time.
    Returns (time_to_window, is_optimal, phase_error_rad, t_total).
    """
    earth_omega = np.sqrt(MU_SUN / r1**3)
    target_omega = np.sqrt(MU_SUN / r2**3)

    _, _, _, t_total, a1, a2, _, _, _, _, _ = calculate_bi_elliptic(r1, r2, rb)

    current_earth_angle = np.arctan2(earth.pos[1], earth.pos[0])
    arrival_time         = current_time + t_total
    target_angle_arrival = (target_omega * arrival_time) % (2 * np.pi)
    # Spacecraft arrives ~180° from launch point
    sc_arrival_angle = current_earth_angle % (2 * np.pi)

    phase_error = abs(sc_arrival_angle - target_angle_arrival)
    phase_error = min(phase_error, 2*np.pi - phase_error)
    is_optimal  = phase_error < 0.05   # ~3°

    time_to_window = phase_error / target_omega if phase_error > 0 else 0

    return time_to_window, is_optimal, phase_error, t_total

def execute_bi_elliptic_burn1(sc, target, rb):
    """
    Apply the first bi-elliptic burn to the spacecraft.
    Sets sc.vel to the correct transfer ellipse 1 velocity.
    Populates sc.transfer_data for subsequent burns.
    """
    r1 = np.linalg.norm(sc.pos - sun.pos)
    r2 = target.a

    dv1, dv2, dv3, t_total, a1, a2, v_at_rb_on_a1, v_at_rb_on_a2, v_at_r2_on_a2, v_c1, v_c2 = \
        calculate_bi_elliptic(r1, r2, rb)

    v_t1 = np.sqrt(MU_SUN * (2/r1 - 1/a1))

    # Tangential direction = perpendicular to radial, in direction of motion
    r_hat = (sc.pos - sun.pos) / r1
    v_tangent = np.array([-r_hat[1], r_hat[0], 0])
    if np.dot(v_tangent, sc.vel) < 0:
        v_tangent = -v_tangent

    sc.vel = v_tangent * v_t1
    acc = compute_acc_sun_only(sc.pos)
    sc.half_vel = sc.vel + 0.5 * acc * BASE_DT

    sc.burns_done = 1
    sc.transfer_type = 'B'
    sc.transfer_data = {
        'dv1': dv1, 'dv2': dv2, 'dv3': dv3,
        't_total': t_total,
        'a1': a1, 'a2': a2,
        'rb': rb, 'r2': r2,
        'target': target,
        'v_at_rb_on_a2': v_at_rb_on_a2,
        'v_circ_target': v_c2,
        'burn2_done': False,
        'burn3_done': False,
    }
    return dv1, dv2, dv3, t_total

def check_and_execute_bi_elliptic_burns(sc):
    """
    Called every frame while sc.transfer_type == 'B'.
    Checks whether the spacecraft is at rb (burn 2) or r2 (burn 3) and fires.
    Uses the same leapfrog integrator pattern as the main code.
    """
    if sc.burns_done == 1:
        sc_r  = np.linalg.norm(sc.pos - sun.pos)
        rb    = sc.transfer_data['rb']
        r_hat = (sc.pos - sun.pos) / sc_r
        radial_vel = np.dot(sc.vel, r_hat)

        if abs(sc_r - rb) < rb * 0.005 and abs(radial_vel) < 100:
            if not sc.transfer_data['burn2_done']:
                v_at_rb_on_a2   = sc.transfer_data['v_at_rb_on_a2']
                v_radial_vec    = radial_vel * r_hat
                v_tangential    = sc.vel - v_radial_vec
                if np.linalg.norm(v_tangential) > 0:
                    v_tangent_dir = v_tangential / np.linalg.norm(v_tangential)
                else:
                    v_tangent_dir = np.array([-r_hat[1], r_hat[0], 0])

                sc.vel = v_tangent_dir * v_at_rb_on_a2
                acc = compute_acc_sun_only(sc.pos)
                sc.half_vel = sc.vel + 0.5 * acc * BASE_DT

                sc.transfer_data['burn2_done'] = True
                sc.burns_done = 2
                dv = abs(np.linalg.norm(v_tangential) - v_at_rb_on_a2)
                print(f"=== BURN 2 at {sc_r/AU:.3f} AU | Δv = {dv/1000:.2f} km/s ===")
                return True

    elif sc.burns_done == 2:
        sc_r  = np.linalg.norm(sc.pos - sun.pos)
        r2    = sc.transfer_data['r2']
        r_hat = (sc.pos - sun.pos) / sc_r
        radial_vel = np.dot(sc.vel, r_hat)

        if abs(sc_r - r2) < r2 * 0.05 and abs(radial_vel) < 2000:
            if not sc.transfer_data['burn3_done']:
                v_circ       = np.sqrt(MU_SUN / sc_r)
                v_radial_vec = radial_vel * r_hat
                v_tangential = sc.vel - v_radial_vec
                if np.linalg.norm(v_tangential) > 0:
                    v_tangent_dir = v_tangential / np.linalg.norm(v_tangential)
                else:
                    target = sc.transfer_data.get('target')
                    if target and np.linalg.norm(target.vel) > 0:
                        v_tangent_dir = target.vel / np.linalg.norm(target.vel)
                    else:
                        v_tangent_dir = np.array([-r_hat[1], r_hat[0], 0])

                dv_circ = abs(np.linalg.norm(v_tangential) - v_circ)
                sc.vel = v_tangent_dir * v_circ
                acc = compute_acc_sun_only(sc.pos)
                sc.half_vel = sc.vel + 0.5 * acc * BASE_DT

                sc.transfer_data['burn3_done'] = True
                sc.burns_done = 3
                sc.has_circularized = True
                print(f"=== BURN 3 (circularization) at {sc_r/AU:.3f} AU | Δv = {dv_circ/1000:.2f} km/s ===")
                return True

    return False

# =============================================================================
# --- Drawing helpers ---
# =============================================================================
def draw_gradient_trail(screen, points, color, max_width=3):
    if len(points) < 2:
        return
    num = len(points)
    for i in range(num - 1):
        fade = (i + 1) / num
        col = tuple(int(c * fade) for c in color)
        w = max(1, int(max_width * fade))
        try:
            pygame.draw.line(screen, col, points[i], points[i+1], w)
        except:
            pass

def draw_orbit_ellipse(screen, body, zoom_factor, offset, color, view_mode):
    if not hasattr(body, 'a') or body.a == 0:
        return
    a = body.a
    e = body.e
    b = a * np.sqrt(1 - e**2)
    c = a * e
    points = []
    for angle in np.linspace(0, 2*np.pi, 180):
        x = a * np.cos(angle) - c
        y = b * np.sin(angle)
        pos = np.array([x, y, 0])
        if hasattr(body, 'e') and body.e > 0:
            incl = np.arcsin(body.pos[2] / (np.linalg.norm(body.pos) + 1e-10))
            rot = rotation_matrix_x(incl)
            pos = rot @ pos
        px, py = project(pos, view_mode)
        if view_mode == 1:
            px += offset[0]
            py += offset[1]
        points.append((int(px), int(py)))
    if len(points) > 1:
        for i in range(0, len(points)-1, 2):
            pygame.draw.line(screen, color, points[i], points[i+1], 1)

def draw_circle_orbit(screen, radius_m, zoom_factor, offset, color, view_mode, width=1):
    """Draw a plain circular orbit of radius_m metres."""
    pts = []
    for angle in np.linspace(0, 2*np.pi, 180):
        pos = np.array([radius_m * np.cos(angle), radius_m * np.sin(angle), 0])
        px, py = project(pos, view_mode)
        if view_mode == 1:
            px += offset[0]
            py += offset[1]
        pts.append((int(px), int(py)))
    if len(pts) > 1:
        for i in range(0, len(pts) - 1, 2):
            pygame.draw.line(screen, color, pts[i], pts[i+1], width)

def draw_hover_info(screen, body, mouse_pos, sun, font):
    speed = np.linalg.norm(body.vel) / 1000
    r_vec = body.pos - sun.pos
    dist_au = np.linalg.norm(r_vec) / AU
    a_au = getattr(body, 'a', 0) / AU
    e = getattr(body, 'e', 0)
    period_yr = np.sqrt(a_au**3) if a_au > 0 else 0
    lines = [
        body.name,
        f"Mass: {body.mass:.2e} kg",
        f"Speed: {speed:.2f} km/s",
        f"Distance: {dist_au:.2f} AU",
        f"Semi-major axis: {a_au:.2f} AU",
        f"Eccentricity: {e:.3f}",
        f"Orbital period: {period_yr:.2f} yr",
    ]
    padding = 5
    w = max(font.size(l)[0] for l in lines) + 2*padding
    h = len(lines) * (font.get_height() + 2) + 2*padding
    bx, by = mouse_pos[0] + 10, mouse_pos[1] + 10
    if bx + w > WIDTH:
        bx = mouse_pos[0] - w - 10
    if by + h > HEIGHT:
        by = mouse_pos[1] - h - 10
    rect = pygame.Rect(bx, by, w, h)
    pygame.draw.rect(screen, (30,30,30), rect)
    pygame.draw.rect(screen, (255,255,255), rect, 1)
    for i, line in enumerate(lines):
        screen.blit(font.render(line, True, (255,255,255)),
                    (rect.x+padding, rect.y+padding+i*(font.get_height()+2)))

# =============================================================================
# --- Pygame setup ---
# =============================================================================
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Solar System - Hohmann & Bi-Elliptic Transfers")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)
small_font = pygame.font.SysFont(None, 18)
running = True

# --- State ---
zoom_factor = 4.5e12 / (WIDTH/2 * 0.9)
zoom_step = 1.1
offset = [0, 0]
dragging = False
drag_start = (0, 0)
EARTH_MOON_SCALE = (moon_dist * 2) / WIDTH
earth_moon_zoom = 1.0
EARTH_DISPLAY_RADIUS = 10
view_mode = 1
total_time = 0.0
paused = False
timewarp = 1

# Planet creation
creating_planet = False
new_planet_start = None
new_planet_radius = 1
planet_counter = 1
created_planets = []

# Spaceflight planning
sf_mode = False
sf_target = mars
launch_requested = False
launch_time = None

# Transfer mode & rb
transfer_mode = 'H'     # 'H' = Hohmann, 'B' = Bi-elliptic
rb_au = 10.0            # Intermediate radius for bi-elliptic (AU)

def project(pos, mode, center=None):
    if center is not None:
        pos = pos - center
    if mode == 3:
        return WIDTH/2 + pos[1]/zoom_factor, HEIGHT/2 + pos[2]/zoom_factor
    return WIDTH/2 + pos[0]/zoom_factor, HEIGHT/2 + pos[1]/zoom_factor

def spawn_spacecraft():
    sc = Spacecraft("Ares", 1000,
                    earth.pos.copy(),
                    earth.vel.copy(),
                    (0, 255, 255), 3)
    sc.timestep_group = 1
    sc.dt = BASE_DT
    sc.next_update_time = total_time
    acc = compute_acc_sun_only(sc.pos)
    sc.half_vel = sc.vel + 0.5 * acc * BASE_DT
    sc.target_planet = None
    sc.has_circularized = False
    sc.transfer_type = transfer_mode
    sc.rb = rb_au * AU
    sc.rb_au = rb_au
    return sc

def reset_spacecraft():
    global spacecraft, launch_time
    spacecraft = spawn_spacecraft()
    spacecraft.burns_done = 0
    spacecraft.has_circularized = False
    launch_time = None
    status_msg_ref[0] = "Spacecraft reset to Earth."

status_msg_ref = ["Press SPACE to launch when ready."]

# =============================================================================
# --- Main loop ---
# =============================================================================
while running:
    clock.tick(1000)

    mu = G * sun.mass

    # --- Pre-compute launch window info for HUD ---
    r1_earth = np.linalg.norm(earth.pos - sun.pos)
    r2_target = sf_target.a

    # Hohmann phase info
    current_phase  = get_phase_angle(earth, sf_target)
    ideal_phase_H  = calculate_phase_angle_for_transfer(r1_earth, r2_target, mu)
    _, _, dv1_H, dv2_H, t_H = calculate_hohmann_transfer(r1_earth, r2_target, mu)[1:]
    # fix unpacking (returns 6 values)
    _, v_t1_H, v_t2_H, dv1_H, dv2_H, t_H = calculate_hohmann_transfer(r1_earth, r2_target, mu)
    phase_diff_H   = abs(current_phase - ideal_phase_H)
    phase_diff_H   = min(phase_diff_H, 360 - phase_diff_H)

    # Bi-elliptic phase info
    rb = rb_au * AU
    dv1_B, dv2_B, dv3_B, t_B, *_ = calculate_bi_elliptic(r1_earth, r2_target, rb)
    time_to_window_B, is_optimal_B, phase_error_B, _ = get_bi_elliptic_phase_info(
        r1_earth, r2_target, rb, total_time)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:

            # --- View modes (outside sf_mode) ---
            if not sf_mode and event.key == pygame.K_1:
                view_mode = 1
            elif not sf_mode and event.key == pygame.K_2:
                view_mode = 2
            elif not sf_mode and event.key == pygame.K_3:
                view_mode = 3

            # --- Target selection (spaceflight mode) ---
            elif sf_mode and event.key in TARGET_MAP:
                sf_target = TARGET_MAP[event.key]
                if spacecraft:
                    spacecraft.target_planet = sf_target
                status_msg_ref[0] = f"Target: {sf_target.name} (e={sf_target.e:.3f})"

            # --- Transfer mode toggle (B key) ---
            elif sf_mode and event.key == pygame.K_b:
                transfer_mode = 'B' if transfer_mode == 'H' else 'H'
                status_msg_ref[0] = f"Transfer mode: {'Bi-Elliptic' if transfer_mode == 'B' else 'Hohmann'}"
                # Reset so spacecraft gets the new mode on next spawn
                if spacecraft and spacecraft.burns_done == 0:
                    spacecraft.transfer_type = transfer_mode
                    spacecraft.rb = rb_au * AU
                    spacecraft.rb_au = rb_au

            # --- rb adjustment (bi-elliptic only, UP/DOWN) ---
            elif sf_mode and transfer_mode == 'B' and event.key == pygame.K_UP:
                rb_au = min(200.0, rb_au + 1.0)
                if spacecraft:
                    spacecraft.rb = rb_au * AU
                    spacecraft.rb_au = rb_au
                status_msg_ref[0] = f"rb = {rb_au:.1f} AU"
            elif sf_mode and transfer_mode == 'B' and event.key == pygame.K_DOWN:
                rb_au = max(max(r1_earth/AU, r2_target/AU) + 0.1, rb_au - 1.0)
                if spacecraft:
                    spacecraft.rb = rb_au * AU
                    spacecraft.rb_au = rb_au
                status_msg_ref[0] = f"rb = {rb_au:.1f} AU"

            # --- Timewarp ---
            elif event.key == pygame.K_q:
                if sf_mode:
                    timewarp = min(50, timewarp + 1)
            elif event.key == pygame.K_w:
                if sf_mode:
                    timewarp = max(1, timewarp - 1)

            # --- Zoom ---
            elif event.key in [pygame.K_EQUALS, pygame.K_PLUS, pygame.K_PAGEUP]:
                if view_mode == 2:
                    earth_moon_zoom *= 1.2
                else:
                    zoom_factor /= zoom_step
            elif event.key in [pygame.K_MINUS, pygame.K_UNDERSCORE, pygame.K_PAGEDOWN]:
                if view_mode == 2:
                    earth_moon_zoom /= 1.2
                else:
                    zoom_factor *= zoom_step

            # --- Undo planet ---
            elif event.key == pygame.K_u and created_planets:
                last = created_planets.pop()
                if last in all_bodies:
                    all_bodies.remove(last)

            # --- Pause / Launch ---
            elif event.key == pygame.K_SPACE:
                if sf_mode:
                    launch_requested = True
                else:
                    paused = not paused

            # --- Asteroid toggle ---
            elif event.key == pygame.K_f:
                asteroids_on = not asteroids_on
                if asteroids_on:
                    generate_asteroids(asteroid_count)
                else:
                    asteroids = []
                all_bodies = bodies + asteroids

            # --- Spaceflight mode toggle ---
            elif event.key == pygame.K_s:
                sf_mode = not sf_mode
                if sf_mode:
                    spacecraft = spawn_spacecraft()
                    spacecraft.target_planet = sf_target
                    status_msg_ref[0] = (f"Spaceflight mode ON. Target: {sf_target.name}  "
                                         f"| Mode: {'Bi-Elliptic' if transfer_mode=='B' else 'Hohmann'}")
                else:
                    spacecraft = None
                    timewarp = 1
                    status_msg_ref[0] = ""

            # --- Reset spacecraft ---
            elif sf_mode and event.key == pygame.K_n:
                reset_spacecraft()
                launch_requested = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mods = pygame.key.get_mods()
            if event.button == 1:
                if mods & pygame.KMOD_SHIFT:
                    creating_planet = True
                    new_planet_start = event.pos
                    new_planet_radius = 1
                elif view_mode == 1:
                    dragging = True
                    drag_start = event.pos

        elif event.type == pygame.MOUSEBUTTONUP:
            if creating_planet:
                x, y = new_planet_start
                pos = np.array([(x - WIDTH/2 - offset[0]) * zoom_factor,
                                (y - HEIGHT/2 - offset[1]) * zoom_factor, 0])
                r_vec = pos - sun.pos
                r_mag = np.linalg.norm(r_vec)

                if r_mag != 0:
                    vel_dir = np.cross([0,0,1], r_vec)
                    vel_dir /= np.linalg.norm(vel_dir)
                    vel = vel_dir * np.sqrt(G * sun.mass / r_mag)
                else:
                    vel = np.zeros(3)

                r_au = r_mag / AU

                if r_au < 2.5:
                    density_factor = 1.0      # rocky
                elif r_au < 10:
                    density_factor = 0.24     # gas
                else:
                    density_factor = 0.27     # ice

                mass = (new_planet_radius / 5.0)**3 * 5.972e24 * density_factor

                color = (np.random.randint(50,255), np.random.randint(50,255), np.random.randint(50,255))
                name = f"Planet {planet_counter}"
                planet_counter += 1
                np_ = Body(name, mass, pos, vel, color, new_planet_radius)
                np_.a = r_mag
                np_.e = 0
                np_.timestep_group = assign_timestep_group(np_, sun.pos, sun.mass)
                np_.dt = BASE_DT * GROUP_MULTIPLIERS[np_.timestep_group]
                np_.next_update_time = total_time
                acc = compute_acceleration_for_body(np_, all_bodies)
                np_.half_vel = np_.vel + 0.5 * acc * np_.dt
                all_bodies.append(np_)
                created_planets.append(np_)
                np_.start_angle = np.arctan2(np_.pos[1], np_.pos[0])
                np_.start_time = total_time
                np_.orbit_checked = False
                a_m = r_mag
                T_seconds = 2 * np.pi * np.sqrt(a_m**3 / (G * sun.mass))
                T_days = T_seconds / DAY_SECONDS
                T_years = T_days / 365.25
                creating_planet = False
            dragging = False

        elif event.type == pygame.MOUSEMOTION:
            if dragging and view_mode == 1:
                dx, dy = event.pos[0]-drag_start[0], event.pos[1]-drag_start[1]
                offset[0] += dx
                offset[1] += dy
                drag_start = event.pos
            elif creating_planet:
                dx, dy = event.pos[0]-new_planet_start[0], event.pos[1]-new_planet_start[1]
                new_planet_radius = max(1, int(np.sqrt(dx**2+dy**2)/10))

        elif event.type == pygame.MOUSEWHEEL:
            if view_mode == 2:
                earth_moon_zoom *= 1.2 if event.y > 0 else 1/1.2
            else:
                zoom_factor /= zoom_step if event.y > 0 else 1/zoom_step

    # =========================================================================
    # --- Physics update ---
    # =========================================================================
    if not paused:
        effective_timewarp = timewarp if sf_mode else 1
        update_bodies(total_time, effective_timewarp)
        total_time += BASE_DT * effective_timewarp

        for body in created_planets:
            if not body.orbit_checked:
                current_angle = np.arctan2(body.pos[1], body.pos[0])
                start_angle = body.start_angle
                angle_diff = (current_angle - start_angle) % (2 * np.pi)
                if angle_diff > 2 * np.pi * 0.99:
                    actual_period = (total_time - body.start_time) / DAY_SECONDS
                    print(f"{body.name} actual simulated period: {actual_period:.1f} days")
                    body.orbit_checked = True

        # --- Spaceflight logic ---
        if sf_mode and spacecraft is not None:

            # ---- Hohmann path (unchanged from original) ----
            if spacecraft.transfer_type == 'H':

                if spacecraft.burns_done == 0:
                    spacecraft.pos = earth.pos.copy()
                    spacecraft.vel = earth.vel.copy()
                    acc = compute_acc_sun_only(spacecraft.pos)
                    spacecraft.half_vel = spacecraft.vel + 0.5 * acc * BASE_DT
                    spacecraft.has_circularized = False

                    if launch_requested:
                        r1 = np.linalg.norm(earth.pos - sun.pos)
                        r2 = sf_target.a
                        a_transfer, v_t1, v_t2, dv1, dv2, t_transfer = calculate_hohmann_transfer(r1, r2, mu)
                        v_earth_dir = spacecraft.vel / np.linalg.norm(spacecraft.vel)
                        spacecraft.vel = v_earth_dir * v_t1
                        acc = compute_acc_sun_only(spacecraft.pos)
                        spacecraft.half_vel = spacecraft.vel + 0.5 * acc * BASE_DT
                        spacecraft.burns_done = 1
                        launch_requested = False
                        launch_time = total_time
                        status_msg_ref[0] = (f"LAUNCHED (Hohmann) to {sf_target.name}! "
                                              f"Δv={dv1/1000:.2f} km/s | ETA: {t_transfer/DAY_SECONDS:.1f} days")

                elif spacecraft.burns_done == 1:
                    update_spacecraft(spacecraft, effective_timewarp)
                    sc_r = np.linalg.norm(spacecraft.pos - sun.pos)
                    target_r = sf_target.a
                    r_hat = (spacecraft.pos - sun.pos) / sc_r
                    radial_vel = np.dot(spacecraft.vel, r_hat)

                    if abs(sc_r - target_r) / target_r < 0.05 and abs(radial_vel) < 2000 \
                            and not spacecraft.has_circularized:
                        v_circ = np.sqrt(mu / sc_r)
                        v_radial_vec = radial_vel * r_hat
                        v_tangential = spacecraft.vel - v_radial_vec
                        if np.linalg.norm(v_tangential) > 0:
                            v_tangent_dir = v_tangential / np.linalg.norm(v_tangential)
                        else:
                            v_tangent_dir = sf_target.vel / np.linalg.norm(sf_target.vel)
                        dv_circ = abs(np.linalg.norm(v_tangential) - v_circ)
                        spacecraft.vel = v_tangent_dir * v_circ
                        acc = compute_acc_sun_only(spacecraft.pos)
                        spacecraft.half_vel = spacecraft.vel + 0.5 * acc * BASE_DT
                        spacecraft.burns_done = 2
                        spacecraft.has_circularized = True
                        status_msg_ref[0] = (f"CIRCULARIZED at {sf_target.name} orbit! "
                                              f"Δv={dv_circ/1000:.2f} km/s | {sc_r/AU:.3f} AU")

                    distance_to_planet = np.linalg.norm(spacecraft.pos - sf_target.pos)
                    if distance_to_planet < 3e8 and spacecraft.burns_done == 2:
                        spacecraft.vel = sf_target.vel.copy()
                        acc = compute_acc_sun_only(spacecraft.pos)
                        spacecraft.half_vel = spacecraft.vel + 0.5 * acc * BASE_DT
                        status_msg_ref[0] = f"✓ CAPTURED by {sf_target.name}! Dist: {distance_to_planet/1e6:.0f} km"
                        spacecraft.name = f"Ares (at {sf_target.name})"

                elif spacecraft.burns_done == 2:
                    update_spacecraft(spacecraft, effective_timewarp)
                    sc_r = np.linalg.norm(spacecraft.pos - sun.pos)
                    v_circ_desired = np.sqrt(mu / sc_r)
                    current_speed = np.linalg.norm(spacecraft.vel)
                    if abs(current_speed - v_circ_desired) / v_circ_desired > 0.01:
                        r_hat = (spacecraft.pos - sun.pos) / sc_r
                        radial_vel = np.dot(spacecraft.vel, r_hat)
                        v_tangential = spacecraft.vel - radial_vel * r_hat
                        if np.linalg.norm(v_tangential) > 0:
                            v_tangent_dir = v_tangential / np.linalg.norm(v_tangential)
                            spacecraft.vel = v_tangent_dir * v_circ_desired
                            acc = compute_acc_sun_only(spacecraft.pos)
                            spacecraft.half_vel = spacecraft.vel + 0.5 * acc * BASE_DT

            # ---- Bi-elliptic path ----
            elif spacecraft.transfer_type == 'B':

                if spacecraft.burns_done == 0:
                    # Park on Earth until launch
                    spacecraft.pos = earth.pos.copy()
                    spacecraft.vel = earth.vel.copy()
                    acc = compute_acc_sun_only(spacecraft.pos)
                    spacecraft.half_vel = spacecraft.vel + 0.5 * acc * BASE_DT
                    spacecraft.has_circularized = False

                    if launch_requested:
                        spacecraft.rb = rb_au * AU
                        spacecraft.rb_au = rb_au
                        dv1, dv2, dv3, t_total = execute_bi_elliptic_burn1(
                            spacecraft, sf_target, spacecraft.rb)
                        launch_requested = False
                        launch_time = total_time

                        if is_optimal_B:
                            window_note = "OPTIMAL WINDOW"
                        else:
                            phase_deg = phase_error_B * 180 / np.pi
                            window_note = f"phase err {phase_deg:.1f}°"

                        status_msg_ref[0] = (
                            f"LAUNCHED (Bi-Elliptic) to {sf_target.name}! "
                            f"rb={rb_au:.1f} AU | {window_note} | "
                            f"Δv={( dv1+dv2+dv3)/1000:.2f} km/s | ETA: {t_total/DAY_SECONDS:.0f} days")
                        print(f"\n=== BI-ELLIPTIC LAUNCH ===")
                        print(f"  Burn 1 Δv : {dv1/1000:.2f} km/s")
                        print(f"  Burn 2 Δv : {dv2/1000:.2f} km/s")
                        print(f"  Burn 3 Δv : {dv3/1000:.2f} km/s")
                        print(f"  Total Δv  : {(dv1+dv2+dv3)/1000:.2f} km/s")
                        print(f"  ETA       : {t_total/DAY_SECONDS:.0f} days")

                elif spacecraft.burns_done in (1, 2):
                    update_spacecraft(spacecraft, effective_timewarp)
                    check_and_execute_bi_elliptic_burns(spacecraft)
                    if spacecraft.burns_done == 2 and not spacecraft.transfer_data.get('burn3_done'):
                        # Still coasting to r2 - update status
                        sc_r = np.linalg.norm(spacecraft.pos - sun.pos)
                        status_msg_ref[0] = (
                            f"Coasting to {sf_target.name} orbit... "
                            f"{sc_r/AU:.3f} AU / {sf_target.a/AU:.3f} AU")

                elif spacecraft.burns_done == 3:
                    update_spacecraft(spacecraft, effective_timewarp)
                    # Maintain circular orbit
                    sc_r = np.linalg.norm(spacecraft.pos - sun.pos)
                    v_circ_desired = np.sqrt(mu / sc_r)
                    current_speed = np.linalg.norm(spacecraft.vel)
                    if abs(current_speed - v_circ_desired) / v_circ_desired > 0.01:
                        r_hat = (spacecraft.pos - sun.pos) / sc_r
                        radial_vel = np.dot(spacecraft.vel, r_hat)
                        v_tangential = spacecraft.vel - radial_vel * r_hat
                        if np.linalg.norm(v_tangential) > 0:
                            v_tangent_dir = v_tangential / np.linalg.norm(v_tangential)
                            spacecraft.vel = v_tangent_dir * v_circ_desired
                            acc = compute_acc_sun_only(spacecraft.pos)
                            spacecraft.half_vel = spacecraft.vel + 0.5 * acc * BASE_DT

    total_days = total_time / DAY_SECONDS

    # =========================================================================
    # --- Drawing ---
    # =========================================================================
    screen.fill((0, 0, 0))

    if creating_planet and new_planet_start:
        pygame.draw.circle(screen, (0,255,0), new_planet_start, new_planet_radius, 2)

    mouse_pos = pygame.mouse.get_pos()

    if view_mode in [1, 3]:
        draw_list = all_bodies[:]
        if sf_mode and spacecraft:
            draw_list.append(spacecraft)

        for body in draw_list:
            px, py = project(body.pos, view_mode)
            if view_mode == 1:
                px += offset[0]
                py += offset[1]

            if len(body.traj) > 1 and (body.mass > 1e20 or "Ares" in body.name):
                trail_pts = [project(p, view_mode) for p in body.traj]
                if view_mode == 1:
                    trail_pts = [(x+offset[0], y+offset[1]) for x,y in trail_pts]
                draw_gradient_trail(screen, trail_pts, body.color)

            draw_r = max(1, int(body.radius))
            if body.name == "Sun":
                draw_r = max(4, int(body.radius / zoom_factor * 1e9))
            pygame.draw.circle(screen, body.color, (int(px), int(py)), draw_r)

            if body.mass > 1e20 or "Ares" in body.name:
                screen.blit(small_font.render(body.name, True, body.color),
                            (int(px)+draw_r+2, int(py)-10))

            dist_mouse = np.linalg.norm(np.array([px,py]) - np.array(mouse_pos))
            if dist_mouse < max(10, draw_r):
                draw_hover_info(screen, body, (px, py), sun, font)

        # Draw orbit ellipses in spaceflight mode
        if sf_mode:
            for body in planet_list:
                faint = tuple(max(0, c // 4) for c in body.color)
                draw_orbit_ellipse(screen, body, zoom_factor, offset, faint, view_mode)
            # Draw target orbit brighter
            target_color = tuple(min(255, c + 100) for c in sf_target.color)
            draw_orbit_ellipse(screen, sf_target, zoom_factor, offset, target_color, view_mode)

            # Draw rb ring (bi-elliptic only) in purple
            if transfer_mode == 'B':
                rb_m = rb_au * AU
                # Only draw if rb is visible (within zoom range)
                if rb_m / zoom_factor < WIDTH * 3:
                    draw_circle_orbit(screen, rb_m, zoom_factor, offset,
                                      (180, 0, 255), view_mode, width=1)
                    # Label the rb ring
                    rb_label_pos = np.array([rb_m, 0, 0])
                    lx, ly = project(rb_label_pos, view_mode)
                    if view_mode == 1:
                        lx += offset[0]
                        ly += offset[1]
                    if 0 < lx < WIDTH and 0 < ly < HEIGHT:
                        screen.blit(small_font.render(f"rb={rb_au:.1f}AU", True, (180, 0, 255)),
                                    (int(lx)+4, int(ly)-10))

        # Target highlight
        if sf_mode and sf_target:
            tpx, tpy = project(sf_target.pos, view_mode)
            if view_mode == 1:
                tpx += offset[0]
                tpy += offset[1]
            pygame.draw.circle(screen, (255,255,0), (int(tpx), int(tpy)),
                               sf_target.radius + 4, 2)

    else:
        center = earth.pos.copy()
        scale = EARTH_MOON_SCALE / earth_moon_zoom
        pts = [(WIDTH/2+(p[0]-center[0])/scale, HEIGHT/2+(p[1]-center[1])/scale)
               for p in earth.traj]
        draw_gradient_trail(screen, pts, earth.color, max_width=4)
        pts = [(WIDTH/2+rel[0]/scale, HEIGHT/2+rel[1]/scale)
               for rel in moon.earth_relative_traj]
        draw_gradient_trail(screen, pts, moon.color, max_width=4)
        for body in [earth, moon]:
            x = int(WIDTH/2 + (body.pos[0]-center[0])/scale)
            y = int(HEIGHT/2 + (body.pos[1]-center[1])/scale)
            r = max(8, min(int(EARTH_DISPLAY_RADIUS*earth_moon_zoom), 40)) \
                if body.name == "Earth" else max(2, min(int(6*earth_moon_zoom), 20))
            pygame.draw.circle(screen, body.color, (x, y), r)

    # =========================================================================
    # --- HUD ---
    # =========================================================================
    years = int(total_days // 365.25)
    months = int((total_days % 365.25) // 30.44)
    days = int((total_days % 365.25) % 30.44)
    screen.blit(font.render(f"Time: {years}y {months}m {days}d", True, (255,255,255)), (10, 10))

    if sf_mode:
        mode_label = "BI-ELLIPTIC" if transfer_mode == 'B' else "HOHMANN"
        mode_color = (180, 0, 255) if transfer_mode == 'B' else (0, 255, 255)

        screen.blit(font.render(
            f"[SPACEFLIGHT - {mode_label}] Target: {sf_target.name}",
            True, mode_color), (10, 35))

        hud_y = 55

        if transfer_mode == 'H':
            # --- Hohmann HUD (unchanged) ---
            phase_color = (0,255,0) if phase_diff_H < 5 else (255,255,0) if phase_diff_H < 10 else (255,0,0)
            screen.blit(font.render(
                f"Phase: {current_phase:.1f}°  |  Ideal: {ideal_phase_H:.1f}°",
                True, phase_color), (10, hud_y)); hud_y += 20
            screen.blit(font.render(
                f"dv1: {dv1_H/1000:.2f} km/s  dv2: {dv2_H/1000:.2f} km/s  "
                f"Total: {(dv1_H+dv2_H)/1000:.2f} km/s",
                True, (255,255,0)), (10, hud_y)); hud_y += 20
            if launch_time is None:
                screen.blit(font.render(
                    f"Transfer time: {t_H/DAY_SECONDS:.1f} days",
                    True, (255,255,255)), (10, hud_y)); hud_y += 20
            else:
                elapsed = (total_time - launch_time) / DAY_SECONDS
                screen.blit(font.render(
                    f"Elapsed: {elapsed:.1f} / {t_H/DAY_SECONDS:.1f} days",
                    True, (255,255,255)), (10, hud_y)); hud_y += 20

        else:
            # --- Bi-elliptic HUD ---
            phase_deg_B = phase_error_B * 180 / np.pi
            if phase_deg_B < 10:
                pe_color = (0,255,0)
                pe_label = "EXCELLENT"
            elif phase_deg_B < 30:
                pe_color = (255,255,0)
                pe_label = "GOOD"
            else:
                pe_color = (255,80,80)
                pe_label = "POOR"

            screen.blit(font.render(
                f"Phase err: {phase_deg_B:.1f}° ({pe_label})  |  "
                f"Window in: {time_to_window_B/DAY_SECONDS:.1f} days",
                True, pe_color), (10, hud_y)); hud_y += 20

            total_dv_B = dv1_B + dv2_B + dv3_B
            screen.blit(font.render(
                f"dv1:{dv1_B/1000:.2f}  dv2:{dv2_B/1000:.2f}  dv3:{dv3_B/1000:.2f}  "
                f"Total:{total_dv_B/1000:.2f} km/s",
                True, (255,255,0)), (10, hud_y)); hud_y += 20

            if launch_time is None:
                screen.blit(font.render(
                    f"Transfer time: {t_B/DAY_SECONDS:.0f} days  |  rb = {rb_au:.1f} AU  "
                    f"(UP/DOWN to adjust)",
                    True, (255,255,255)), (10, hud_y)); hud_y += 20
            else:
                elapsed = (total_time - launch_time) / DAY_SECONDS
                screen.blit(font.render(
                    f"Elapsed: {elapsed:.0f} / {t_B/DAY_SECONDS:.0f} days  |  rb = {rb_au:.1f} AU",
                    True, (255,255,255)), (10, hud_y)); hud_y += 20

            # Burn progress (only after launch)
            if spacecraft and spacecraft.transfer_type == 'B' and spacecraft.burns_done >= 1:
                burn_labels = {1: "Coasting to rb", 2: "Coasting to target orbit", 3: "✓ Circularized"}
                burn_text = burn_labels.get(spacecraft.burns_done, "")
                screen.blit(font.render(f"Burn status: {burn_text}", True, mode_color),
                            (10, hud_y)); hud_y += 20

            # Efficiency note
            ratio = r2_target / r1_earth
            if ratio > 11.94:
                screen.blit(small_font.render(
                    "✓ Bi-elliptic more efficient than Hohmann for this target",
                    True, (100,255,100)), (10, hud_y)); hud_y += 16
            elif ratio > 5:
                screen.blit(small_font.render(
                    "Bi-elliptic comparable to Hohmann for this target",
                    True, (200,200,100)), (10, hud_y)); hud_y += 16

        # Common controls line
        screen.blit(font.render(
            "Timewarp: Q/W  |  1-7: target  |  B: toggle mode  |  N: reset  |  S: exit  |  Space: launch",
            True, (180,180,180)), (10, hud_y)); hud_y += 20

        screen.blit(font.render(status_msg_ref[0], True, (255,200,0)), (10, hud_y)); hud_y += 20

        if spacecraft and spacecraft.burns_done >= 1:
            sc_dist = np.linalg.norm(spacecraft.pos - sun.pos) / AU
            screen.blit(font.render(
                f"SC: {sc_dist:.3f} AU  |  Target orbit: {sf_target.a/AU:.3f} AU",
                True, (0,255,255)), (10, hud_y))

        # Launch window indicator (Hohmann)
        if transfer_mode == 'H' and launch_time is None and phase_diff_H < 5:
            pygame.draw.rect(screen, (0,255,0), (WIDTH//2-180, HEIGHT-40, 360, 30))
            screen.blit(font.render("LAUNCH WINDOW OPEN - PRESS SPACE", True, (0,0,0)),
                        (WIDTH//2-150, HEIGHT-35))

        # Launch window indicator (Bi-elliptic)
        if transfer_mode == 'B' and launch_time is None and is_optimal_B:
            pulse = abs(np.sin(total_time / DAY_SECONDS * 0.3))
            box_color = (int(80 + 175*pulse), 0, 255)
            pygame.draw.rect(screen, box_color, (WIDTH//2-210, HEIGHT-40, 420, 30))
            screen.blit(font.render("BI-ELLIPTIC WINDOW OPEN - PRESS SPACE", True, (255,255,255)),
                        (WIDTH//2-195, HEIGHT-35))

    else:
        screen.blit(font.render(
            "S: Spaceflight mode | F: Asteroids | Space: Pause | U: Undo planet",
            True, (180,180,180)), (10, 35))
        screen.blit(font.render(
            "Q: Speed up | W: Slow down | 1/2/3: Views | Shift+drag: Add planet",
            True, (180,180,180)), (10, 55))

    if paused:
        screen.blit(font.render("PAUSED", True, (255,0,0)), (WIDTH//2-40, 10))

    pygame.display.flip()

pygame.quit()