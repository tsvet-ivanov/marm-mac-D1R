#VANILLA RING MODEL ARCHITECTURE

# Import necessary packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import math
from scipy.special import erf

# Parameters
num_triangles = 48
num_circles = 6
radius = 1
triangle_size = 0.05
circle_radius = 0.05
scale_factor = 10  # Adjust this value as needed

# Multiply all coordinates and sizes by scale_factor
radius *= scale_factor
triangle_size *= scale_factor
circle_radius *= scale_factor

theta_triangles = np.linspace(0, 2. * np.pi, num_triangles, endpoint=False)
theta_circles = np.linspace(0, 2. * np.pi, num_circles, endpoint=False) - np.pi / 2
offset = 1

# Create a custom colormap
colors = [(0.8, 1, 0.8), (0.4, 1, 0.4), (0, 0.6, 0)]  # R, G, B
cmap_name = 'myGreens'
cmap_green = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=256)

# Create the figure and axis
fig, ax = plt.subplots(figsize=(24, 24))

# Create the outline rings
circle_ring_triangles = plt.Circle((0, 0), radius, edgecolor="#E41A1C", facecolor="none", linewidth=5, zorder=4)
circle_ring_outer_circles = plt.Circle((0, 0), 0.33 * radius, edgecolor="#4DAF4A", facecolor="none", linewidth=5, zorder=1)  # RGBA color, alpha = 1
circle_ring_inner_circles = plt.Circle((0, 0), 0.66 * radius, edgecolor="#377EB8", facecolor="none", linewidth=5, zorder=1)  # RGBA color, alpha = 1
ax.add_artist(circle_ring_triangles)
ax.add_artist(circle_ring_outer_circles)
ax.add_artist(circle_ring_inner_circles)

# Create the triangles
for i, angle in enumerate(theta_triangles):
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    triangle = patches.RegularPolygon((x, y), numVertices=3, radius=triangle_size,
                                      orientation=angle + np.pi / 2, color="#E41A1C", zorder=5)
    ax.add_patch(triangle)

# Create the blue circles
for angle in theta_circles:
    x = 0.66 * radius * np.cos(angle)
    y = 0.66 * radius * np.sin(angle)
    circle = patches.Circle((x, y), radius=circle_radius, color="#377EB8", zorder=2)
    ax.add_patch(circle)

# Create the green circles
for i, angle in enumerate(theta_circles):
    x = 0.33 * radius * np.cos(angle)
    y = 0.33 * radius * np.sin(angle)
    circle = patches.Circle((x, y), radius=circle_radius, color="#4DAF4A", zorder=3)
    ax.add_patch(circle)

# Finalize the plot
ax.set_aspect("equal")
ax.set_xlim(-radius - offset, radius + offset)
ax.set_ylim(-radius - offset, radius + offset)
ax.axis("off")
plt.savefig('/Users/nd23721/Documents/marm-mac/vanila_ring.png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()

#GREEN I_OPP POPULATION SCHEMATIC

# Import necessary packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import math
from scipy.special import erf

# Parameters
sigma_weight_profile_I2E = 14.4
Jpos_inhib2excit = 1.6
num_triangles = 48
num_circles = 6
radius = 1
triangle_size = 0.05
circle_radius = 0.05
scale_factor = 10  # Adjust this value as needed

# Multiply all coordinates and sizes by scale_factor
radius *= scale_factor
triangle_size *= scale_factor
circle_radius *= scale_factor

theta_triangles = np.linspace(0, 2. * np.pi, num_triangles, endpoint=False)
theta_circles = np.linspace(0, 2. * np.pi, num_circles, endpoint=False) - np.pi / 2
offset = 1
selected_circle = 3

# Adjust the weights for the inhibitory connections
tmp_tuned_inhib2excit = math.sqrt(2. * math.pi) * sigma_weight_profile_I2E * erf(180. / math.sqrt(2.) / sigma_weight_profile_I2E) / 360.
Jneg_tuned_inhib2excit = (1. - Jpos_inhib2excit * tmp_tuned_inhib2excit) / (1. - tmp_tuned_inhib2excit)
shift_size = 3*num_triangles // 4
presyn_tuned_inhib2excit_weight_kernel = np.array([
    Jneg_tuned_inhib2excit + (Jpos_inhib2excit - Jneg_tuned_inhib2excit) *
    np.exp(-.5 * ((360. * min((i+shift_size) % num_triangles, num_triangles - ((i+shift_size) % num_triangles)) / num_triangles) ** 2) / sigma_weight_profile_I2E ** 2)
    for i in range(num_triangles)
])

presyn_tuned_inhib2excit_weight_kernel_adjusted = presyn_tuned_inhib2excit_weight_kernel + 0.011  # Adjust the constant as needed

# Make sure the values stay within the range [0, 1]
presyn_tuned_inhib2excit_weight_kernel_adjusted = np.clip(presyn_tuned_inhib2excit_weight_kernel_adjusted, 0, 1)

# Normalize the kernel to the range [0, 1]
kernel_min = presyn_tuned_inhib2excit_weight_kernel_adjusted.min()
kernel_max = presyn_tuned_inhib2excit_weight_kernel_adjusted.max()
presyn_tuned_inhib2excit_weight_kernel_adjusted_normalized = (presyn_tuned_inhib2excit_weight_kernel_adjusted - kernel_min) / (kernel_max - kernel_min)

presyn_tuned_inhib2excit_weight_kernel_adjusted_normalized = presyn_tuned_inhib2excit_weight_kernel_adjusted_normalized + 0.2

presyn_tuned_inhib2excit_weight_kernel = np.clip(presyn_tuned_inhib2excit_weight_kernel_adjusted_normalized, 0, 1)

# Create a custom colormap
colors = [(0.8, 1, 0.8), (0.4, 1, 0.4), (0, 0.6, 0)]  # R, G, B
cmap_name = 'myGreens'
cmap_green = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=256)

# Create the figure and axis
fig, ax = plt.subplots(figsize=(24, 24))

# Create the outline rings
circle_ring_triangles = plt.Circle((0, 0), radius, edgecolor="#E41A1C", facecolor="none", linewidth=5, zorder=4)
circle_ring_outer_circles = plt.Circle((0, 0), 0.33 * radius, edgecolor="#4DAF4A", facecolor="none", linewidth=5, zorder=1)  # RGBA color, alpha = 1
circle_ring_inner_circles = plt.Circle((0, 0), 0.66 * radius, edgecolor="#377EB8", facecolor="none", linewidth=5, zorder=1)  # RGBA color, alpha = 1
ax.add_artist(circle_ring_triangles)
ax.add_artist(circle_ring_outer_circles)
ax.add_artist(circle_ring_inner_circles)

# Create the triangles
for i, angle in enumerate(theta_triangles):
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    triangle = patches.RegularPolygon((x, y), numVertices=3, radius=triangle_size,
                                      orientation=angle + np.pi / 2, color="#E41A1C", zorder=5)
    ax.add_patch(triangle)

# Create the blue circles
for angle in theta_circles:
    x = 0.66 * radius * np.cos(angle)
    y = 0.66 * radius * np.sin(angle)
    circle = patches.Circle((x, y), radius=circle_radius, color="#377EB8", zorder=2)
    ax.add_patch(circle)

# Create the green circles with dimmed overlays
circle_radius_dimmed = circle_radius + 0.01
for i, angle in enumerate(theta_circles):
    x = 0.33 * radius * np.cos(angle)
    y = 0.33 * radius * np.sin(angle)
    zorder_circle = 5 if i == selected_circle else 1  # changed zorder for non-selected circles
    circle = patches.Circle((x, y), radius=circle_radius, color="#4DAF4A", zorder=zorder_circle)
    ax.add_patch(circle)

# Dim the green circles and the black circle on which they are positioned
overlay = patches.Circle((0, 0), 0.66 * radius + circle_radius + 0.002, color='white', alpha=0.9, zorder=2)
ax.add_patch(overlay)

# Create the aura for the selected circle
# Generate a radial gradient image
gradient_size = 100  # Adjust this to change the resolution of the gradient
gradient = np.zeros((gradient_size, gradient_size, 4))  # The third dimension is for RGBA
for y in range(gradient_size):
    for x in range(gradient_size):
        # Calculate the distance to the center of the image
        distance_to_center = np.sqrt((x - gradient_size / 2) ** 2 + (y - gradient_size / 2) ** 2)
        # Calculate the alpha value based on the distance
        alpha = np.exp(
            -5 * distance_to_center / gradient_size)  # The constant 5 controls how quickly the aura disperses
        # Only set the alpha value inside the circle
        if distance_to_center <= gradient_size / 2:
            gradient[y, x] = np.array([77/255, 175/255, 74/255, alpha])  # Set the color to green
# Define the position of the selected circle
x_circle_green = 0.33 * radius * np.cos(theta_circles[selected_circle])
y_circle_green = 0.33 * radius * np.sin(theta_circles[selected_circle])
# Display the gradient as an image with the selected circle at the center
aura_radius = 0.1 * scale_factor
ax.imshow(gradient, extent=(x_circle_green - aura_radius, x_circle_green + aura_radius,
                            y_circle_green - aura_radius, y_circle_green + aura_radius), zorder=3)

# Plot triangles and connections with updated colors
for i, angle in enumerate(theta_triangles):
    x_triangle = radius * np.cos(angle)
    y_triangle = radius * np.sin(angle)
    x_start = 0.33 * radius * np.cos(theta_circles[selected_circle])
    y_start = 0.33 * radius * np.sin(theta_circles[selected_circle])
    shifted_index = (i + num_triangles // 2) % num_triangles
    # Scale the linewidth to be between 1 and 5 based on the kernel weight
    linewidth = 1 + 10 * presyn_tuned_inhib2excit_weight_kernel[shifted_index]
    ax.plot([x_start, x_triangle], [y_start, y_triangle], color="#4DAF4A", alpha=1, zorder=4, linewidth=linewidth)  # set zorder to 4

# Finalize the plot
ax.set_aspect("equal")
ax.set_xlim(-radius - offset, radius + offset)
ax.set_ylim(-radius - offset, radius + offset)
ax.axis("off")
plt.savefig('/Users/nd23721/Documents/marm-mac/green_ring.png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()


#RED E POPULATION SCHEMATIC

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.path as path
import math
from scipy.special import erf

# Parameters
num_triangles = 48
num_circles = 6
radius = 1
triangle_size = 0.05
circle_radius = 0.05
scale_factor = 10  # Adjust this value as needed

# Multiply all coordinates and sizes by scale_factor
radius *= scale_factor
triangle_size *= scale_factor
circle_radius *= scale_factor

theta_triangles = np.linspace(0, 2. * np.pi, num_triangles, endpoint=False)
theta_circles = np.linspace(0, 2. * np.pi, num_circles, endpoint=False) - np.pi / 2
offset = 1

# Create a custom colormap
colors = [(0.8, 1, 0.8), (0.4, 1, 0.4), (0, 0.6, 0)]  # R, G, B
cmap_name = 'myGreens'
cmap_green = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=256)

# Code from Script 1 for calculating the weight kernel
sigma_weight_profile_E2E = 14.4
Jpos_excit2excit = 2.1
selected_circle = 3

tmp_excit2excit = math.sqrt(2. * math.pi) * sigma_weight_profile_E2E * erf(180. / math.sqrt(2.) / sigma_weight_profile_E2E) / 360.
Jneg_excit2excit = (1. - Jpos_excit2excit * tmp_excit2excit) / (1. - tmp_excit2excit)
presyn_excit2excit_weight_kernel = np.array([
    Jneg_excit2excit + (Jpos_excit2excit - Jneg_excit2excit) *
    np.exp(-.5 * ((360. * min(j, num_triangles - j) / num_triangles) ** 2) / sigma_weight_profile_E2E ** 2)
    for j in range(num_triangles)
])

# Original kernel
presyn_excit2excit_weight_kernel_normalized = (presyn_excit2excit_weight_kernel - np.min(presyn_excit2excit_weight_kernel)) / \
                                   (np.max(presyn_excit2excit_weight_kernel) - np.min(presyn_excit2excit_weight_kernel))

shift_amount = num_triangles // 4
presyn_excit2excit_weight_kernel_shifted = np.roll(presyn_excit2excit_weight_kernel, shift_amount)

# Add a constant to every value
presyn_excit2excit_weight_kernel_adjusted = presyn_excit2excit_weight_kernel_shifted + 0.011

# Make sure the values stay within the range [0, 1]
presyn_excit2excit_weight_kernel_adjusted_clipped = np.clip(presyn_excit2excit_weight_kernel_adjusted, 0, 1)

# Normalize the kernel to the range [0, 1]
kernel_min = presyn_excit2excit_weight_kernel_adjusted_clipped.min()
kernel_max = presyn_excit2excit_weight_kernel_adjusted_clipped.max()
presyn_excit2excit_weight_kernel_adjusted_normalized = (presyn_excit2excit_weight_kernel_adjusted_clipped - kernel_min) / (kernel_max - kernel_min)

presyn_excit2excit_weight_kernel_adjusted_normalized_adjusted = presyn_excit2excit_weight_kernel_adjusted_normalized + 0.2

presyn_excit2excit_weight_kernel_adjusted_normalized =  np.clip(presyn_excit2excit_weight_kernel_adjusted_normalized_adjusted, 0, 1)

# Calculate the position of the starting triangle
start_triangle_index = (num_triangles // num_circles * selected_circle - num_triangles // 4) % num_triangles
x_start_triangle = radius * np.cos(theta_triangles[start_triangle_index])
y_start_triangle = radius * np.sin(theta_triangles[start_triangle_index])

# Create the figure and axis
fig, ax = plt.subplots(figsize=(24, 24))

# Create the outline rings
circle_ring_triangles = plt.Circle((0, 0), radius, edgecolor="#E41A1C", facecolor="none", linewidth=5, zorder=5)
circle_ring_outer_circles = plt.Circle((0, 0), 0.33 * radius, edgecolor="#4DAF4A", facecolor="none", linewidth=5, zorder=2)  # RGBA color, alpha = 1
circle_ring_inner_circles = plt.Circle((0, 0), 0.66 * radius, edgecolor="#377EB8", facecolor="none", linewidth=5, zorder=3)  # RGBA color, alpha = 1
ax.add_artist(circle_ring_triangles)
ax.add_artist(circle_ring_outer_circles)
ax.add_artist(circle_ring_inner_circles)

# Create the triangles
for i, angle in enumerate(theta_triangles):
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    triangle = patches.RegularPolygon((x, y), numVertices=3, radius=triangle_size,
                                      orientation=angle + np.pi / 2, color="#E41A1C", zorder=5)
    ax.add_patch(triangle)

# Create the blue circles
for angle in theta_circles:
    x = 0.66 * radius * np.cos(angle)
    y = 0.66 * radius * np.sin(angle)
    circle = patches.Circle((x, y), radius=circle_radius, color="#377EB8", zorder=3)
    ax.add_patch(circle)

# Create the green circles
for i, angle in enumerate(theta_circles):
    x = 0.33 * radius * np.cos(angle)
    y = 0.33 * radius * np.sin(angle)
    circle = patches.Circle((x, y), radius=circle_radius, color="#4DAF4A", zorder=2)
    ax.add_patch(circle)

# Draw lines from the starting triangle to other triangles, color based on weight kernel
for i, angle in enumerate(theta_triangles):
    x_triangle = radius * np.cos(angle)
    y_triangle = radius * np.sin(angle)
    # Scale the linewidth to be between 1 and 5 based on the kernel weight
    linewidth = 1 + 10 * presyn_excit2excit_weight_kernel_adjusted_normalized[i]
    control_point = (0, 0.33 * scale_factor)
    bezier_path = path.Path([(x_start_triangle, y_start_triangle),
                             control_point,
                             (x_triangle, y_triangle)],
                            [path.Path.MOVETO, path.Path.CURVE3, path.Path.CURVE3])
    patch = patches.PathPatch(bezier_path, facecolor='none', edgecolor="#E41A1C", linewidth=linewidth, alpha=1, zorder=5)
    ax.add_patch(patch)

# Create the aura
gradient_size = 100  # Adjust this to change the resolution of the gradient
gradient = np.zeros((gradient_size, gradient_size, 4))  # The third dimension is for RGBA
for y in range(gradient_size):
    for x in range(gradient_size):
        # Calculate the distance to the center of the image
        distance_to_center = np.sqrt((x - gradient_size / 2) ** 2 + (y - gradient_size / 2) ** 2)
        # Calculate the alpha value based on the distance
        alpha = np.exp(-3 * distance_to_center / gradient_size)  # The constant 5 controls how quickly the aura fades
        # Only set the alpha value inside the circle
        if distance_to_center <= gradient_size / 2:
            gradient[y, x] = np.array([0.88, 0.1, 0.1, alpha])  # Set the color to red
# Display the gradient as an image with the selected triangle at the center
aura_radius = 0.075 * scale_factor  # Adjust this to change the size of the aura
ax.imshow(gradient, extent=(x_start_triangle - aura_radius, x_start_triangle + aura_radius,
                            y_start_triangle - aura_radius, y_start_triangle + aura_radius), zorder=5)

# Add the dimming effect (overlay)
overlay = patches.Circle((0, 0), radius+1, color='white', alpha=0.7, zorder=4)
ax.add_patch(overlay)

# Finalize the plot
ax.set_aspect("equal")
ax.set_xlim(-radius - offset, radius + offset)
ax.set_ylim(-radius - offset, radius + offset)
ax.axis("off")
plt.savefig('/Users/nd23721/Documents/marm-mac/red_ring.png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()

# BLUE I_NEAR POPULATION SCHEMATIC

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
import random
import math
from scipy.special import erf
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Define the parameters
sigma_weight_profile_I2E = 14.4
Jpos_inhib2excit = 1.6
num_triangles = 48
num_circles = 6
radius = 1
triangle_size = 0.05
circle_radius = 0.05

scale_factor = 10  # Adjust this value as needed

# Multiply all coordinates and sizes by scale_factor
radius *= scale_factor
triangle_size *= scale_factor
circle_radius *= scale_factor

# Calculate the angles for the triangles and circles
theta_triangles = np.linspace(0, 2. * np.pi, num_triangles, endpoint=False)
theta_circles = np.linspace(0, 2. * np.pi, num_circles,
                            endpoint=False) - np.pi / 2  # Subtract pi/2 to start from the top

selected_circle = 3

offset = 1

# precompute the weight profile for the TUNED INHIBITORY recurrent population
tmp_tuned_inhib2excit = math.sqrt(2. * math.pi) * sigma_weight_profile_I2E * erf(180. / math.sqrt(2.) / sigma_weight_profile_I2E) / 360.
Jneg_tuned_inhib2excit = (1. - Jpos_inhib2excit * tmp_tuned_inhib2excit) / (1. - tmp_tuned_inhib2excit)
shift_size = 3*num_triangles // 4
presyn_tuned_inhib2excit_weight_kernel = np.array([
    Jneg_tuned_inhib2excit + (Jpos_inhib2excit - Jneg_tuned_inhib2excit) *
    np.exp(-.5 * ((360. * min((i+shift_size) % num_triangles, num_triangles - ((i+shift_size) % num_triangles)) / num_triangles) ** 2) / sigma_weight_profile_I2E ** 2)
    for i in range(num_triangles)
])

# Adjust the kernel to make both stronger and weaker connections darker
# Add a constant to every value
presyn_tuned_inhib2excit_weight_kernel_adjusted = presyn_tuned_inhib2excit_weight_kernel + 0.011  # Adjust the constant as needed

# Make sure the values stay within the range [0, 1]
presyn_tuned_inhib2excit_weight_kernel_adjusted = np.clip(presyn_tuned_inhib2excit_weight_kernel_adjusted, 0, 1)

# Normalize the kernel to the range [0, 1]
kernel_min = presyn_tuned_inhib2excit_weight_kernel_adjusted.min()
kernel_max = presyn_tuned_inhib2excit_weight_kernel_adjusted.max()
presyn_tuned_inhib2excit_weight_kernel_adjusted_normalized = (presyn_tuned_inhib2excit_weight_kernel_adjusted - kernel_min) / (kernel_max - kernel_min)

presyn_tuned_inhib2excit_weight_kernel_adjusted_normalized = presyn_tuned_inhib2excit_weight_kernel_adjusted_normalized + 0.2

presyn_tuned_inhib2excit_weight_kernel = np.clip(presyn_tuned_inhib2excit_weight_kernel_adjusted_normalized, 0, 1)


# Create the figure and axis with a larger figure size
fig, ax = plt.subplots(figsize=(24, 24))

# Calculate the position of the selected circle
x_circle = 0.66 * radius * np.cos(theta_circles[selected_circle])
y_circle = 0.66 * radius * np.sin(theta_circles[selected_circle])

# Shift offset in radians
shift_offset = (360 / num_triangles) / 2 * (np.pi / 180)

# Radius of the small circle (to be added to the starting position of the lines)
small_circle_radius = 0.05

# Create the aura for the selected circle
# Generate a radial gradient image
gradient_size = 100  # Adjust this to change the resolution of the gradient
gradient = np.zeros((gradient_size, gradient_size, 4))  # The third dimension is for RGBA
for y in range(gradient_size):
    for x in range(gradient_size):
        # Calculate the distance to the center of the image
        distance_to_center = np.sqrt((x - gradient_size / 2) ** 2 + (y - gradient_size / 2) ** 2)
        # Calculate the alpha value based on the distance
        alpha = np.exp(-5 * distance_to_center / gradient_size)  # The constant 5 controls how quickly the aura disperses
        # Only set the alpha value inside the circle
        if distance_to_center <= gradient_size / 2:
            gradient[y, x] = np.array([0, 0, 1, alpha])  # Set the color to blue
# Display the gradient as an image with the selected circle at the center
aura_radius = 0.1
ax.imshow(gradient, extent=(x_circle - aura_radius, x_circle + aura_radius,
                            y_circle - aura_radius, y_circle + aura_radius), zorder=4)

# Create the outline rings
circle_ring_triangles = plt.Circle((0, 0), radius, edgecolor="#E41A1C", facecolor="none", linewidth=5, zorder=1)
circle_ring_outer_circles = plt.Circle((0, 0), 0.33 * radius, edgecolor="#4DAF4A", facecolor="none", linewidth=5, zorder=1)
circle_ring_inner_circles_blue = plt.Circle((0, 0), 0.66 * radius, edgecolor="#377EB8", facecolor="none", linewidth=5, zorder=1)
ax.add_artist(circle_ring_triangles)
ax.add_artist(circle_ring_outer_circles)
ax.add_artist(circle_ring_inner_circles_blue)

# Create the blue circles (except the selected one)
blue_circles = []
for i, angle in enumerate(theta_circles):
    x = 0.66 * radius * np.cos(angle)
    y = 0.66 * radius * np.sin(angle)
    circle = patches.Circle((x, y), radius=circle_radius, color='#377EB8', linewidth=2, zorder=2)  # adjust linewidth here
    blue_circles.append(circle)
    ax.add_patch(circle)

# Create the green circles
green_circles = []
for angle in theta_circles:
    x = 0.33 * radius * np.cos(angle)
    y = 0.33 * radius * np.sin(angle)
    circle = patches.Circle((x, y), radius=circle_radius, color='#4DAF4A', linewidth=2, zorder=2)  # adjust linewidth here
    green_circles.append(circle)
    ax.add_patch(circle)

# Dim the green circles and the black circle on which they are positioned
overlay = patches.Circle((0, 0), 0.66 * radius + circle_radius + 0.002, color='white', alpha=0.7, zorder=3)
ax.add_patch(overlay)

# Create the triangles and draw lines projecting from the selected circle
triangles = []
lines = []
for i, angle in enumerate(theta_triangles):
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    triangle = patches.RegularPolygon(
        (x, y),
        numVertices=3,
        radius=triangle_size,
        orientation=angle + np.pi / 2,
        color="#E41A1C",
        zorder=5  # Set the zorder to 5 to plot on top
    )
    triangles.append(triangle)
    ax.add_patch(triangle)
    # Draw lines projecting from the selected circle, with color based on the kernel value
    x_start = x_circle + small_circle_radius / 2 * np.cos(theta_circles[selected_circle])
    y_start = y_circle + small_circle_radius / 2 * np.sin(theta_circles[selected_circle])
    linewidth = 1 + 10 * presyn_tuned_inhib2excit_weight_kernel_adjusted_normalized[i]
    line = ax.plot([x_start, x], [y_start, y], color="#377EB8", alpha=1, linewidth=linewidth, zorder=4)  # Set the zorder to 4 to plot on top, adjust linewidth here
    lines.append(line)

# Create the selected blue circle
x = 0.66 * radius * np.cos(theta_circles[selected_circle])
y = 0.66 * radius * np.sin(theta_circles[selected_circle])
circle = patches.Circle((x, y), radius=circle_radius, color="#377EB8", linewidth=2, zorder=6)  # Set the zorder to 6 to plot on top, adjust linewidth here
ax.add_patch(circle)

# Make sure the aspect ratio of the plot is equal
ax.set_aspect("equal")
# Set the limits of the plot
ax.set_xlim(-radius - 1, radius + 1)
ax.set_ylim(-radius - 1, radius + 1)
# Remove the axes for a cleaner look
ax.axis("off")

plt.savefig('/Users/nd23721/Documents/marm-mac/blue_ring.png', dpi=300, bbox_inches='tight', pad_inches=0)

# Show the plot
plt.show()
# Save the figure with a high DPI
# fig.savefig("ring.png", dpi=300)