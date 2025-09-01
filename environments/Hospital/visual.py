from environments.Hospital.players import *
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import random
import imageio.v2 as imageio
from matplotlib.lines import Line2D


def render_simulation(self, gif_filename="hospital_simulation_v3.gif", save_gif=True):
    """
    Generates a GIF animation of the simulation run
    """
    
    if not save_gif:
        print("GIF rendering is skipped as per configuration.")
        return
    output_path = pathlib.Path("visual_output")
    output_path.mkdir(exist_ok=True)
    temp_dir = output_path / "frames"
    temp_dir.mkdir(exist_ok=True)
    images = []

    print(f"Rendering enhanced simulation frames into '{temp_dir}'...")

    PATIENT_COLOURS = {"high": "#d62728", "medium": "#ff7f0e", "low": "#1f77b4"}
    NURSE_COLOURS = {"idle": "#2ca02c", "moving": "#98df8a", "on_duty": "#176017"}
    ROBOT_COLOURS = {"idle": "#7f7f7f", "moving": "#c7c7c7", "on_duty": "#525252"}
    ROOM_COLOURS = {"low": "#d1e7dd", "medium": "#fff3cd", "high": "#f8d7da"}
    PATIENT_STATUS_MARKERS = {
        "waiting_for_escort": "H", "being_treated": "P",
        "waiting_for_exit": "X", "exited": "*", "default": "o"
    }

    num_frames_to_render = len(self.snapshots) // 4
    if num_frames_to_render < 1: num_frames_to_render = 1
    frame_skip = len(self.snapshots) // num_frames_to_render

    for i, snapshot in enumerate(self.snapshots):
        if i % frame_skip != 0: continue

        fig, ax = plt.subplots(figsize=(16, 10))
        fig.patch.set_facecolor('#f0f0f0')
        ax.set_facecolor('#ffffff')

        agent_positions = snapshot["positions"]
        agent_destinations = snapshot.get("destinations", {})
        
        active_patients = [p for p in self.patients if p.id in agent_positions]
        waiting_patients = len([p for p in active_patients if "waiting" in p.status])
        treating_patients = len([p for p in active_patients if p.status == "being_treated"])
        idle_nurses = len([n for n in self.nurses if n.id in agent_positions and n.role == 'idle'])
        
        info_text = (
            f"Time: {snapshot['time']:03d} mins ({snapshot['time']/60:.1f} hrs)\n"
            f"--- PATIENTS ---\n"
            f"Active: {len(active_patients)}\n"
            f"Waiting: {waiting_patients}\n"
            f"In Treatment: {treating_patients}\n"
            f"--- STAFF ---\n"
            f"Nurses Idle: {idle_nurses}"
        )
        
        fig.text(0.01, 0.98, info_text, transform=ax.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        patient_counts = {room: 0 for room in self.room_positions}
        for p in active_patients:
            if isinstance(p.position, str) and p.position in patient_counts:
                patient_counts[p.position] += 1
        
        for start, end in self.room_travel_times.keys():
            pos_start, pos_end = self.get_position(start), self.get_position(end)
            if pos_start and pos_end:
                 ax.plot([pos_start[0], pos_end[0]], [pos_start[1], pos_end[1]], color="lightgray", linestyle="--", linewidth=1, zorder=0)

        for room, pos in self.room_positions.items():
            count = patient_counts.get(room, 0)
            color = ROOM_COLOURS['low'] if count == 0 else (ROOM_COLOURS['medium'] if count < 4 else ROOM_COLOURS['high'])
            ax.scatter(*pos, marker="s", s=1800, color=color, edgecolor="black", linewidth=1.5, alpha=0.6, zorder=1)
            ax.text(pos[0], pos[1], f"{room}\n({count})", ha='center', va='center', fontsize=9, fontweight='bold', zorder=2)


        all_agents_to_draw = [a for a in (self.patients + self.nurses + self.robots) if a.id in agent_positions]
        all_agents_to_draw.sort(key=lambda x: 1 if isinstance(x, Patient) else 0)

        for agent in all_agents_to_draw:
            base_pos = agent_positions[agent.id]
            x, y = base_pos

            if isinstance(base_pos, str):
                room_center_x, room_center_y = self.get_position(base_pos)
                x, y = room_center_x + random.uniform(-0.6, 0.6), room_center_y + random.uniform(-0.6, 0.6)

            if agent.id in agent_destinations:
                dest_pos = self.get_position(agent_destinations[agent.id])
                if dest_pos:
                    ax.arrow(x, y, dest_pos[0] - x, dest_pos[1] - y,
                             color='black', alpha=0.4, width=0.05,
                             head_width=0.3, length_includes_head=True, zorder=2)
            
            if isinstance(agent, Patient):
                if agent.true_priority == 'high':
                    ax.scatter(x, y, s=500, color='#e5ff00', alpha=0.6, zorder=2.5)
                
                colour = PATIENT_COLOURS.get(agent.true_priority, "blue")
                marker = PATIENT_STATUS_MARKERS.get(agent.status, PATIENT_STATUS_MARKERS["default"])
                ax.scatter(x, y, color=colour, s=120, marker=marker, zorder=3, ec='black', lw=0.5)
                ax.text(x, y - 0.35, f"P{agent.id}", ha="center", fontsize=8, zorder=4)

            elif isinstance(agent, Nurse):
                colour = NURSE_COLOURS.get(agent.role, "green")
                ax.scatter(x, y, color=colour, s=150, marker="^", zorder=2, ec='black', lw=0.5)
                ax.text(x, y + 0.35, f"N{agent.id}", ha="center", fontsize=8, zorder=4)

            elif isinstance(agent, Robot):
                colour = ROBOT_COLOURS.get(agent.role, "black")
                ax.scatter(x, y, color=colour, s=150, marker="s", zorder=2, ec='black', lw=0.5)
                ax.text(x, y - 0.35, f"R{agent.id}", ha="center", fontsize=8, zorder=4)
        
        fig.suptitle(f"Hospital Simulation", fontsize=16, fontweight='bold')
        
        legend_elements = [
            # Patients
            Line2D([0], [0], marker='o', color='w', label='Patient (High Prio)', markerfacecolor=PATIENT_COLOURS['high'], markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Patient (Medium Prio)', markerfacecolor=PATIENT_COLOURS['medium'], markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Patient (Low Prio)', markerfacecolor=PATIENT_COLOURS['low'], markersize=10),
            # Patient Status
            Line2D([0], [0], marker='H', color='w', label='Status: Waiting Escort', markerfacecolor='grey', markersize=10),
            Line2D([0], [0], marker='P', color='w', label='Status: In Treatment', markerfacecolor='grey', markersize=10),
            # Staff
            Line2D([0], [0], marker='^', color='w', label='Nurse (Idle)', markerfacecolor=NURSE_COLOURS['idle'], markersize=10),
            Line2D([0], [0], marker='^', color='w', label='Nurse (Moving)', markerfacecolor=NURSE_COLOURS['moving'], markersize=10),
            Line2D([0], [0], marker='s', color='w', label='Robot (Idle)', markerfacecolor=ROBOT_COLOURS['idle'], markersize=10),
        ]
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

        ax.set_xlim(-1, 9)
        ax.set_ylim(-1, 11)
        ax.set_aspect('equal', adjustable='box')
        ax.axis("off")
        plt.tight_layout(rect=[0, 0, 0.85, 0.95])

        frame_path = temp_dir / f"frame_{i:04d}.png"
        fig.savefig(frame_path, dpi=96)
        images.append(imageio.imread(frame_path))
        plt.close(fig)

    #  Save GIF
    if images:
        final_gif_path = output_path / gif_filename
        print(f"Creating GIF '{final_gif_path}' from {len(images)} frames...")
        imageio.mimsave(final_gif_path, images, fps=10)
        print(f"Simulation GIF saved successfully.")
    else:
        print("No snapshots were recorded, skipping GIF rendering.")

    print("Cleaning up temporary frame files...")
    for img_path in temp_dir.iterdir():
        img_path.unlink()
    temp_dir.rmdir()
                
def debug_tick(self):
    """One-line dashboard every simulated minute."""
    
    # patients
    outside_q = len(self.entrance_queue)
    waiting_in = len([p for p in self.rooms["Entrance"] if isinstance(p, Patient) and p.status == "waiting"])
    in_triage = len([p for p in self.rooms["Triage"] if isinstance(p, Patient) and p.status == "accepted"])
    treating = len([p for p in self.patients if p.status == "being_treated"])
    done_so_far = len(self.completed_patients)

    # staff 
    idle_nurse = len([n for n in self.nurses  if getattr(n, "role", "idle") == "idle"])
    move_nurse = len([n for n in self.nurses  if getattr(n, "role", "") == "moving"])
    idle_robot = len([r for r in self.robots  if getattr(r, "role", "idle") == "idle"])
    move_robot = len([r for r in self.robots  if getattr(r, "role", "") == "moving"])
    busy_docs = len([d for d in self.doctors if d.busy_until > self.current_time])

    print(f"[t={self.current_time:03}m] "
        f"Q_out={outside_q:2} | waiting_ent={waiting_in:2} | "
        f"in_triage={in_triage:2} | treating={treating:2} | done={done_so_far:2} || "
        f"N(idle/mov)={idle_nurse}/{move_nurse} "
        f"R(idle/mov)={idle_robot}/{move_robot} "
        f"Docs_busy={busy_docs}")

    if outside_q > 0:
        q_ids = [p.id for p in self.entrance_queue]
        print(f" >> Patients in entrance queue: {q_ids}")
        
def plot_arrival_distributions(arrivals_per_day, bins=60, output_filename="arrival_distributions.png"):
    """
    Plots the distribution of patient arrivals for multiple days on a single chart
    Each line = one day's arrival histogram
    """

    plt.figure(figsize=(12, 6))
    plt.style.use('seaborn-v0_8-whitegrid')

    for i, arr_ts in enumerate(arrivals_per_day):
        counts, _ = np.histogram(arr_ts, bins=bins, range=(0, 480))
        plt.plot(counts, label=f"Day {i+1} Simulation", marker='o', markersize=4, linestyle='--')

    plt.xlabel("Time of Day (in 8-minute intervals)")
    plt.ylabel("Number of Patient Arrivals")
    plt.title("Patient Arrival Distributions Across Simulated Days")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"✓ Arrival distribution plot saved to {output_filename}")
