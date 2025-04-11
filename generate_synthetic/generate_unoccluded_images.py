import os
import math
import json
from PIL import Image, ImageDraw
import random

def is_triangular(n):
    """Check if a number is triangular."""
    k = int((math.sqrt(8 * n + 1) - 1) / 2)
    return n == k * (k + 1) // 2

def generate_circle_positions(n, img_size, center_offset=(0, 0)):
    """Generate positions of dots arranged in a circle."""
    radius = img_size * 0.25  # Reduced radius to bring dots closer
    center_x = img_size / 2 + center_offset[0]
    center_y = img_size / 2 + center_offset[1]
    angles = [2 * math.pi * i / n for i in range(n)]
    positions = [
        (
            center_x + radius * math.cos(angle),
            center_y + radius * math.sin(angle)
        ) for angle in angles
    ]
    return positions

def generate_triangle_positions(n, img_size, start_x_offset=0, start_y_offset=0):
    """Generate positions of dots arranged in a triangle."""
    k = int((math.sqrt(8 * n + 1) - 1) / 2)  # Number of rows
    positions = []
    dot_spacing = img_size * 0.4 / k  # Reduced spacing to bring dots closer
    start_x = (img_size - (k * dot_spacing)) / 2 + dot_spacing / 2 + start_x_offset
    y = img_size * 0.3 + start_y_offset  # Adjusted starting Y position
    for row in range(1, k + 1):
        x = start_x + (k - row) * (dot_spacing / 2)
        for _ in range(row):
            positions.append((x, y))
            x += dot_spacing
        y += dot_spacing * math.sqrt(3) / 2
    return positions

def is_rectangular(n):
    """Check if a number can form a rectangle (product of two integers)."""
    return n >= 1  # All positive integers can form a rectangle

def generate_rectangle_positions(n, img_size, start_x_offset=0, start_y_offset=0):
    """Generate positions of dots arranged in a filled rectangle."""
    factors = [i for i in range(1, n + 1) if n % i == 0]
    rows = factors[len(factors) // 2]
    cols = n // rows

    positions = []

    if cols == 1:
        # Handle the case of a single column
        dot_spacing_y = img_size * 0.8 / (rows - 1) if rows > 1 else 0
        start_x = img_size / 2 + start_x_offset
        start_y = (img_size - (rows - 1) * dot_spacing_y) / 2 + start_y_offset

        positions = [(start_x, start_y + i * dot_spacing_y) for i in range(rows)]
    elif rows == 1:
        # Handle the case of a single row
        dot_spacing_x = img_size * 0.8 / (cols - 1) if cols > 1 else 0
        start_x = (img_size - (cols - 1) * dot_spacing_x) / 2 + start_x_offset
        start_y = img_size / 2 + start_y_offset

        positions = [(start_x + i * dot_spacing_x, start_y) for i in range(cols)]
    else:
        # Handle the general case
        dot_spacing_x = img_size * 0.2 / (cols - 1)
        dot_spacing_y = img_size * 0.4 / (rows - 1)
        start_x = (img_size - (cols - 1) * dot_spacing_x) / 2 + start_x_offset
        start_y = (img_size - (rows - 1) * dot_spacing_y) / 2 + start_y_offset

        positions = [
            (start_x + col * dot_spacing_x, start_y + row * dot_spacing_y)
            for row in range(rows)
            for col in range(cols)
        ]

    return positions

def circle_intersects_rectangle(x_c, y_c, r, x0, y0, x1, y1):
    """Check if a circle intersects a rectangle."""
    x_min = min(x0, x1)
    x_max = max(x0, x1)
    y_min = min(y0, y1)
    y_max = max(y0, y1)
    x_closest = max(x_min, min(x_c, x_max))
    y_closest = max(y_min, min(y_c, y_max))
    distance_sq = (x_c - x_closest) ** 2 + (y_c - y_closest) ** 2
    return distance_sq <= r ** 2

def rectangle_intersects_rectangle(rect1, rect2):
    """Check if two rectangles intersect. Each rect is defined as (x0, y0, x1, y1)."""
    x0_1, y0_1, x1_1, y1_1 = rect1
    x0_2, y0_2, x1_2, y1_2 = rect2
    # Rectangles intersect if they overlap
    return not (x1_1 < x0_2 or x1_2 < x0_1 or y1_1 < y0_2 or y1_2 < y0_1)

def add_occluder(draw, positions, dot_radius, shape):
    """Add a black box that occludes at least one dot and compute the number of dots occluded."""
    # Choose a dot to occlude
    occluded_dot = random.choice(positions)
    # Define the rectangle size
    rect_size = dot_radius * random.uniform(6, 12)  # Increased size
    rect_half = rect_size / 2
    # Rectangle coordinates
    x0 = occluded_dot[0] - rect_half
    y0 = occluded_dot[1] - rect_half
    x1 = occluded_dot[0] + rect_half
    y1 = occluded_dot[1] + rect_half
    draw.rectangle([x0, y0, x1, y1], fill="black")

    # Compute the number of dots occluded
    num_occluded = 0
    for pos in positions:
        if shape == "circle":
            if circle_intersects_rectangle(pos[0], pos[1], dot_radius, x0, y0, x1, y1):
                num_occluded += 1
        elif shape == "square":
            dot_rect = (pos[0] - dot_radius, pos[1] - dot_radius, pos[0] + dot_radius, pos[1] + dot_radius)
            occluder_rect = (x0, y0, x1, y1)
            if rectangle_intersects_rectangle(dot_rect, occluder_rect):
                num_occluded += 1
    return num_occluded

def main():
    output_dir = "unoccluded_dataset"
    os.makedirs(output_dir, exist_ok=True)
    metadata = []
    img_size = 200
    dot_radius = 5

    shift_amount = img_size * 0.13  # 16% of the image size
    offsets = [
        (0, 0),  # Center
        (-shift_amount, -shift_amount),  # Top-left
        (shift_amount, -shift_amount),   # Top-right
        (-shift_amount, shift_amount),   # Bottom-left
        (shift_amount, shift_amount)     # Bottom-right
    ]

    colors = ["blue", "green", "red", "orange", "purple"]

    for n in range(5, 16):
        for shape in ["circle", "triangle", "rectangle"]:
            # Check if the shape can be formed with n dots
            if shape == "triangle" and not is_triangular(n):
                continue
            if shape == "rectangle" and not is_rectangular(n):
                continue

            for i, offset in enumerate(offsets):
                for color in colors:
                    # Generate dot positions
                    if shape == "circle":
                        positions = generate_circle_positions(n, img_size, center_offset=offset)
                    elif shape == "triangle":
                        positions = generate_triangle_positions(
                            n, img_size, start_x_offset=offset[0], start_y_offset=offset[1]
                        )
                    elif shape == "rectangle":
                        positions = generate_rectangle_positions(
                            n, img_size, start_x_offset=offset[0], start_y_offset=offset[1]
                        )

                    # ----------------- Generate image with dots (circles) -----------------
                    # Create a blank image
                    image_dots = Image.new("RGB", (img_size, img_size), "white")
                    draw_dots = ImageDraw.Draw(image_dots)

                    # Draw the dots (circles)
                    for pos in positions:
                        x0 = pos[0] - dot_radius
                        y0 = pos[1] - dot_radius
                        x1 = pos[0] + dot_radius
                        y1 = pos[1] + dot_radius
                        draw_dots.ellipse([x0, y0, x1, y1], fill=color)

                    # Add occluder and compute the number of dots occluded
                    num_occluded_dots = 0

                    # Save the image with dots
                    image_name_dots = f"{shape}_{n}_dots_{i}_{color}_dots.png"
                    image_path_dots = os.path.join(output_dir, image_name_dots)
                    image_dots.save(image_path_dots)

                    # Store metadata for dots
                    metadata.append({
                        "image_file": image_name_dots,
                        "ground_truth": n,
                        "shape": shape,
                        "offset": offset,
                        "color": color,
                        "num_occluded": num_occluded_dots,
                        "dot_shape": "circle"
                    })

                    # ----------------- Generate image with boxes (squares) -----------------
                    # Create a blank image
                    image_boxes = Image.new("RGB", (img_size, img_size), "white")
                    draw_boxes = ImageDraw.Draw(image_boxes)

                    # Draw the boxes (squares)
                    for pos in positions:
                        x0 = pos[0] - dot_radius
                        y0 = pos[1] - dot_radius
                        x1 = pos[0] + dot_radius
                        y1 = pos[1] + dot_radius
                        draw_boxes.rectangle([x0, y0, x1, y1], fill=color)

                    # Add occluder and compute the number of dots occluded
                    num_occluded_boxes = 0

                    # Save the image with boxes
                    image_name_boxes = f"{shape}_{n}_dots_{i}_{color}_boxes.png"
                    image_path_boxes = os.path.join(output_dir, image_name_boxes)
                    image_boxes.save(image_path_boxes)

                    # Store metadata for boxes
                    metadata.append({
                        "image_file": image_name_boxes,
                        "ground_truth": n,
                        "shape": shape,
                        "offset": offset,
                        "color": color,
                        "num_occluded": num_occluded_boxes,
                        "dot_shape": "square"
                    })

    # Save metadata to JSON file
    # metadata_path = "unocc_metadata.json"
    # with open(metadata_path, "w") as json_file:
    #     json.dump(metadata, json_file, indent=4)

if __name__ == "__main__":
    main()