import pygame
import math
import random
from typing import List, Tuple, Optional

# Constants
PARTICLE_COUNT = 500
PARTICLE_RADIUS = 4
PARTICLE_SPEED = 5
SPLIT_THRESHOLD = 10
BOUNCE_DAMPING = 0.99
G = 1.0  # Gravitational constant 
BLACK_HOLE_MASS = 400
BLACK_HOLE_RADIUS = 20  # Larger radius for visibility
MIN_BLACK_HOLE_DISTANCE = BLACK_HOLE_RADIUS      # Minimum distance for force calculation with black hole
MIN_PARTICLE_DISTANCE = PARTICLE_RADIUS  # Minimum distance for force calculation between particles

# Get screen info
pygame.init()
screen_info = pygame.display.Info()
WIDTH, HEIGHT = screen_info.current_w, screen_info.current_h

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)  # Color for text display
PURPLE = (128, 0, 128)  # Color for black hole
BLUE = (0, 0, 255)     # Color for slowest particles
RED = (255, 0, 0)      # Color for fastest particles

def get_velocity_color(vx: float, vy: float) -> Tuple[int, int, int]:
    """Calculate particle color based on velocity magnitude."""
    velocity = math.sqrt(vx * vx + vy * vy)
    # Normalize velocity to a value between 0 and 1
    max_velocity = PARTICLE_SPEED * 2  # Adjust this value based on typical velocities
    normalized = min(velocity / max_velocity, 1.0)
    
    # Interpolate between blue (slow) and red (fast)
    r = int(normalized * 255)
    b = int((1 - normalized) * 255)
    g = 0
    return (r, g, b)

class BlackHole:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.mass = BLACK_HOLE_MASS
    
    def draw(self, screen):
        # Create a single surface for both glow and core
        max_glow_radius = int(BLACK_HOLE_RADIUS * 1.5)
        surface_size = max_glow_radius * 2
        glow_surface = pygame.Surface((surface_size, surface_size), pygame.SRCALPHA)
        
        # Draw multiple glow layers with decreasing alpha
        num_layers = 8
        for i in range(num_layers, 0, -1):
            radius = BLACK_HOLE_RADIUS * (1 + (0.5 * i / num_layers))
            alpha = int(255 * (1 - (i / num_layers)) * 0.5)  # 50% max opacity
            pygame.draw.circle(glow_surface, (128, 0, 128, alpha),
                             (max_glow_radius, max_glow_radius), int(radius))
        
        # Draw the core on top
        pygame.draw.circle(glow_surface, PURPLE,
                         (max_glow_radius, max_glow_radius), BLACK_HOLE_RADIUS)
        
        # Blit the entire surface
        screen.blit(glow_surface,
                   (int(self.x - max_glow_radius),
                    int(self.y - max_glow_radius)))

class Particle:
    def __init__(self, x, y, vx, vy, mass):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.mass = mass
        self.is_captured = False  # Track if particle is currently captured

    def apply_force(self, fx, fy):
        self.vx += fx
        self.vy += fy

    def update_position(self):
        self.x += self.vx
        self.y += self.vy

    def draw(self, screen):
        color = get_velocity_color(self.vx, self.vy)
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), PARTICLE_RADIUS)

    def check_boundary_collision(self):
        if self.x < 0 or self.x > WIDTH:
            self.vx *= -BOUNCE_DAMPING
        if self.y < 0 or self.y > HEIGHT:
            self.vy *= -BOUNCE_DAMPING

    def get_kinetic_energy(self) -> float:
        """Calculate the kinetic energy of the particle."""
        velocity_squared = self.vx * self.vx + self.vy * self.vy
        return 0.5 * self.mass * velocity_squared

class QuadTree:
    """QuadTree node for Barnes-Hut algorithm"""
    def __init__(self, x: float, y: float, width: float, height: float):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.particles: List[Particle] = []
        self.children: List[Optional[QuadTree]] = [None] * 4
        self.total_mass = 0
        self.center_of_mass_x = 0
        self.center_of_mass_y = 0
        self.theta = 0.5  # Opening angle threshold

    def insert(self, particle: Particle) -> bool:
        # If particle is not in this quad, don't add it
        if not (self.x <= particle.x < self.x + self.width and
                self.y <= particle.y < self.y + self.height):
            return False

        # Update center of mass and total mass
        self.total_mass += particle.mass
        self.center_of_mass_x = (self.center_of_mass_x * (self.total_mass - particle.mass) +
                                particle.x * particle.mass) / self.total_mass
        self.center_of_mass_y = (self.center_of_mass_y * (self.total_mass - particle.mass) +
                                particle.y * particle.mass) / self.total_mass

        # If this node doesn't have children and has room, add the particle here
        if not self.children[0] and len(self.particles) < 1:
            self.particles.append(particle)
            return True

        # If this node doesn't have children, create them
        if not self.children[0]:
            self.subdivide()
            # Move existing particle to children
            if self.particles:
                self.insert_to_children(self.particles[0])
                self.particles = []

        # Add new particle to children
        return self.insert_to_children(particle)

    def subdivide(self):
        half_width = self.width / 2
        half_height = self.height / 2
        
        self.children[0] = QuadTree(self.x, self.y, half_width, half_height)  # NW
        self.children[1] = QuadTree(self.x + half_width, self.y, half_width, half_height)  # NE
        self.children[2] = QuadTree(self.x, self.y + half_height, half_width, half_height)  # SW
        self.children[3] = QuadTree(self.x + half_width, self.y + half_height, half_width, half_height)  # SE

    def insert_to_children(self, particle: Particle) -> bool:
        for child in self.children:
            if child.insert(particle):
                return True
        return False

    def compute_force(self, particle: Particle) -> tuple[float, float]:
        if not self.total_mass:
            return 0, 0

        # If this node is a leaf with a single particle and it's not the particle we're calculating forces for
        if len(self.particles) == 1 and self.particles[0] is not particle:
            return compute_gravitational_force(particle, self.particles[0])

        # Calculate distance to center of mass
        dx = self.center_of_mass_x - particle.x
        dy = self.center_of_mass_y - particle.y
        distance = math.sqrt(dx * dx + dy * dy)

        # Prevent division by zero and handle very close particles
        if distance < MIN_PARTICLE_DISTANCE:
            return 0, 0

        # If this node is sufficiently far away, use its center of mass
        if self.width / distance < self.theta:
            force = G * (self.total_mass * particle.mass) / (distance * distance)
            return force * dx / distance, force * dy / distance

        # Otherwise, recursively compute forces from children
        total_fx, total_fy = 0, 0
        for child in self.children:
            if child and child.total_mass > 0:
                fx, fy = child.compute_force(particle)
                total_fx += fx
                total_fy += fy
        
        return total_fx, total_fy

    def draw(self, screen):
        """Draw the quadtree structure."""
        # Draw the boundary of this quad
        pygame.draw.rect(screen, (40, 40, 40), (self.x, self.y, self.width, self.height), 1)
        
        # Draw center of mass if this node has mass
        if self.total_mass > 0:
            pygame.draw.circle(screen, (100, 100, 100), 
                             (int(self.center_of_mass_x), int(self.center_of_mass_y)), 2)

        # Recursively draw children
        for child in self.children:
            if child:
                child.draw(screen)

def compute_gravitational_force(p1: Particle, p2: Particle, black_hole: BlackHole = None) -> tuple[float, float]:
    """Compute gravitational force between particles and black hole if provided."""
    total_fx, total_fy = 0, 0
    
    # Force between particles
    if p2 is not None:
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        r = math.sqrt(dx ** 2 + dy ** 2)
        # Apply minimum distance threshold for particle interactions
        r = max(r, MIN_PARTICLE_DISTANCE)
        force = G / (r ** 2)
        total_fx += force * dx / r
        total_fy += force * dy / r
    
    # Force from black hole
    if black_hole is not None:
        dx = black_hole.x - p1.x
        dy = black_hole.y - p1.y
        r = math.sqrt(dx ** 2 + dy ** 2)
        # Apply minimum distance threshold for black hole interactions
        r = max(r, MIN_BLACK_HOLE_DISTANCE)
        force = G * black_hole.mass / (r ** 2)
        total_fx += force * dx / r
        total_fy += force * dy / r
    
    return total_fx, total_fy

def check_collision(p1: Particle, p2: Particle) -> bool:
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    distance = math.sqrt(dx ** 2 + dy ** 2)
    if distance <= PARTICLE_RADIUS * 2:
        return True
    return False

def split_particle(particles: List[Particle], p: Particle) -> None:
    particles.remove(p)
    for _ in range(2):
        angle = random.uniform(0, 2 * math.pi)  # Random angle for initial velocity
        speed = random.uniform(0.5, 1.5) * PARTICLE_SPEED  # Random speed factor
        vx = speed * math.cos(angle)
        vy = speed * math.sin(angle)
        particles.append(Particle(p.x, p.y, vx, vy, p.mass / 2))

def handle_collision(p1: Particle, p2: Particle) -> None:
    # Elastic collision with momentum conservation
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    distance = math.sqrt(dx ** 2 + dy ** 2)
    
    if distance == 0:  # Prevent division by zero
        return
        
    # Normal vector of collision
    nx = dx / distance
    ny = dy / distance
    
    # Relative velocity
    dvx = p2.vx - p1.vx
    dvy = p2.vy - p1.vy
    
    # Relative velocity along normal
    vn = dvx * nx + dvy * ny
    
    # Don't collide if particles are moving apart
    if vn > 0:
        return
        
    # Collision impulse
    j = -(1 + BOUNCE_DAMPING) * vn
    j /= 1/p1.mass + 1/p2.mass
    
    # Apply impulse
    p1.vx -= j * nx / p1.mass
    p1.vy -= j * ny / p1.mass
    p2.vx += j * nx / p2.mass
    p2.vy += j * ny / p2.mass

def main():
    if not pygame.display.get_init():
        print("Failed to initialize PyGame")
        return

    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
    pygame.display.set_caption("Gravitational Particles Simulation")
    clock = pygame.time.Clock()
    
    # Create black hole at center of screen
    black_hole = BlackHole(WIDTH // 2, HEIGHT // 2)
    currently_captured = 0
    show_quadtree = False  # Toggle for quadtree visualization
    
    # Initialize font for FPS counter
    try:
        font = pygame.font.Font(None, 36)
    except pygame.error:
        font = pygame.font.SysFont('arial', 36)
    
    # Initialize particles
    particles = []
    for _ in range(PARTICLE_COUNT):
        # Create particles avoiding the black hole area
        while True:
            x = random.randint(PARTICLE_RADIUS, WIDTH - PARTICLE_RADIUS)
            y = random.randint(PARTICLE_RADIUS, HEIGHT - PARTICLE_RADIUS)
            dx = x - black_hole.x
            dy = y - black_hole.y
            distance = math.sqrt(dx ** 2 + dy ** 2)
            if distance > BLACK_HOLE_RADIUS * 3:  # Keep particles away from black hole initially
                break
        
        # Give particles initial orbital velocity
        angle = math.atan2(dy, dx) + math.pi/2  # Perpendicular to radius
        speed = math.sqrt(G * black_hole.mass / distance) * 0.7  # 70% of escape velocity
        vx = speed * math.cos(angle)
        vy = speed * math.sin(angle)
        
        mass = random.uniform(1, 5)
        particles.append(Particle(x, y, vx, vy, mass))

    running = True
    while running:
        screen.fill(BLACK)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:  # Press ESC to exit
                    running = False
                elif event.key == pygame.K_SPACE:  # Press SPACE to toggle quadtree visualization
                    show_quadtree = not show_quadtree

        # Reset current capture count and rebuild quadtree
        currently_captured = 0
        quad_tree = QuadTree(0, 0, WIDTH, HEIGHT)
        for p in particles:
            quad_tree.insert(p)

        # Draw quadtree if enabled (before particles for better visibility)
        if show_quadtree:
            quad_tree.draw(screen)

        # Update forces using Barnes-Hut algorithm
        for p in particles:
            # Check if particle is captured by black hole
            dx = p.x - black_hole.x
            dy = p.y - black_hole.y
            distance = math.sqrt(dx * dx + dy * dy)
            
            if distance <= BLACK_HOLE_RADIUS:
                currently_captured += 1

            # Force from black hole
            fx, fy = compute_gravitational_force(p, None, black_hole)
            
            # Forces from other particles using quadtree
            tree_fx, tree_fy = quad_tree.compute_force(p)
            
            # Apply combined forces
            p.apply_force(fx + tree_fx, fy + tree_fy)

        # Draw black hole first (background layer)
        black_hole.draw(screen)

        # Update positions and handle collisions
        for p in particles:
            p.update_position()
            p.check_boundary_collision()
            p.draw(screen)

        # Handle collisions and splitting
        for i, p1 in enumerate(particles[:]):  # Create a copy of the list to iterate
            for p2 in particles[i+1:]:
                if check_collision(p1, p2):
                    if p1.mass >= SPLIT_THRESHOLD and p2.mass >= SPLIT_THRESHOLD:
                        split_particle(particles, p1)
                        split_particle(particles, p2)
                    else:
                        handle_collision(p1, p2)

        # Calculate total system energy
        total_energy = sum(p.get_kinetic_energy() for p in particles)
        
        # Draw FPS, particle count, energy, and capture counts
        fps = int(clock.get_fps())
        fps_text = font.render(f'FPS: {fps}', True, YELLOW)
        count_text = font.render(f'Particles: {len(particles)}', True, YELLOW)
        energy_text = font.render(f'TKEnergy: {total_energy:.1f}', True, YELLOW)
        captured_text = font.render(f'Captured: {currently_captured}', True, YELLOW)
        quadtree_text = font.render('Press SPACE to toggle QuadTree view', True, YELLOW)
        
        screen.blit(fps_text, (10, 10))
        screen.blit(count_text, (10, 50))    
        screen.blit(captured_text, (10, 90))
        screen.blit(energy_text, (10, 130))
        screen.blit(quadtree_text, (10, 170))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()

