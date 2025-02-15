import uproot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.mplot3d import Axes3D
import argparse
import sys
from matplotlib.widgets import Button

def plot_event(event_number, tree, metadata=None, ax=None, fig=None):
    if ax is None or fig is None:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
    else:
        # Store current view angles before clearing
        elev, azim = ax.elev, ax.azim
        
        # Remove existing colorbars and clear all axes except buttons
        for ax_obj in fig.axes:
            if not isinstance(ax_obj, plt.matplotlib.axes.Axes) or \
               not any(button in ax_obj.get_label() for button in ['prev_button', 'next_button']):
                ax_obj.remove()
        
        # Create new main axes
        ax = fig.add_subplot(111, projection='3d')
        
        # Restore view angles
        ax.view_init(elev=elev, azim=azim)
    
    # Clear the main axes
    ax.clear()
    
    # Get data for the specific event
    particle_pt = tree['particle_pt'].array()[event_number]
    particle_eta = tree['particle_eta'].array()[event_number]
    particle_phi = tree['particle_phi'].array()[event_number]
    particle_jetIndex = tree['particle_jetIndex'].array()[event_number]
    jet_pt = tree['jet_pt'].array()[event_number]
    jet_eta = tree['jet_eta'].array()[event_number]
    jet_phi = tree['jet_phi'].array()[event_number]
    jet_area = tree['jet_area'].array()[event_number]

    # Define fixed colors for jets (using Set1 which has 9 distinct colors)
    jet_colors = plt.cm.Set1(np.linspace(0, 1, 9))  # Fixed to 9 colors
    
    # Plot particles as vertical lines to their pT
    for eta, phi, pt, jetidx in zip(particle_eta, particle_phi, particle_pt, particle_jetIndex):
        if jetidx >= 0:
            # Jet constituents: use jet color with higher opacity
            color = jet_colors[jetidx % 9]  # Wrap around if more than 9 jets
            alpha = 0.8
        else:
            # Unassociated particles: gray and more transparent
            color = 'gray'
            alpha = 0.3
            
        ax.plot([eta, eta], [phi, phi], [0, pt], 
                color=color,
                alpha=alpha,
                linewidth=2)
    
    # Plot jets and their radius
    R = 0.4  # Anti-kT parameter
    for i, (eta, phi, pt, area) in enumerate(zip(jet_eta, jet_phi, jet_pt, jet_area)):
        # Plot jet marker at the top as an arrow
        ax.scatter([eta], [phi], [pt], 
                  color=jet_colors[i % 9],  # Wrap around if more than 9 jets
                  s=100,
                  marker='^',
                  label=f'Jet {i}: pT={pt:.1f} GeV, area={area:.2f}')
        
        # Draw thicker vertical line for jet
        ax.plot([eta, eta], [phi, phi], [0, pt], 
                color=jet_colors[i % 9],
                linestyle=':',
                linewidth=2,
                alpha=0.5)
        
        # Draw jet radius circle
        theta = np.linspace(0, 2*np.pi, 100)
        circle_eta = eta + R * np.cos(theta)
        circle_phi = phi + R * np.sin(theta)
        circle_z = np.zeros_like(theta)  # Draw circle at z=0
        ax.plot(circle_eta, circle_phi, circle_z, 
                color=jet_colors[i % 9], 
                linestyle='-', 
                alpha=0.8)

    # Add metadata text box if available
    if metadata:
        metadata_text = (
            f"{metadata.get('beamSpecies', 'Unknown collision')}\n"
            f"{metadata.get('collisionEnergy', 'Unknown energy')}\n"
            f"Jet finder: {metadata.get('jetAlgorithm', 'Unknown algorithm')}"
        )
        fig.text(0.02, 0.98, metadata_text,
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='top')

    # Customize plot
    ax.set_xlabel('η')
    ax.set_ylabel('φ')
    ax.set_zlabel('pT (GeV)')
    ax.set_title(f'Event {event_number}')
    
    # Set axis limits
    ax.set_xlim(-7, 7)
    ax.set_ylim(-np.pi, np.pi)
    max_pt = max(max(particle_pt), max(jet_pt) if len(jet_pt) > 0 else 0)
    ax.set_zlim(0, max_pt * 1.2)  # Add 20% margin on top
    
    # Update legend
    if len(jet_pt) > 0:
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='gray', linewidth=2, label='Unassociated particles'),
            Line2D([0], [0], marker='^', color='red', 
                  markersize=10, label=f'Jets (R={R})')
        ]
        # Add specific jet information
        for i, pt in enumerate(jet_pt):
            legend_elements.append(
                Line2D([0], [0], color=jet_colors[i % 9],
                      linewidth=2,
                      label=f'Jet {i}: pT={pt:.1f} GeV')
            )
        ax.legend(handles=legend_elements, loc='upper right')
    
    # Set the view angle
    ax.view_init(elev=20, azim=45)

    return fig, ax

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Plot events from ROOT file')
    parser.add_argument('filename', type=str, help='Path to the ROOT file')
    parser.add_argument('-e', '--event', type=int, default=0, 
                        help='Initial event number to display (default: 0)')
    
    args = parser.parse_args()
    
    try:
        file = uproot.open(args.filename)
        tree = file["events"]
    except FileNotFoundError:
        print(f"Error: Could not find file '{args.filename}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    try:
        metadata = {
            'jetAlgorithm': str(tree['jetAlgorithm'].array()[0]),
            'beamSpecies': str(tree['beamSpecies'].array()[0]),
            'collisionEnergy': str(tree['collisionEnergy'].array()[0])
        }
    except:
        print("Warning: Could not read metadata")
        metadata = None

    n_events = len(tree['particle_pt'].array())
    event_number = args.event

    # Create figure with a specific layout
    fig = plt.figure(figsize=(12, 9))
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.15, top=0.9)
    
    # Create main plotting area with 3D projection
    main_ax = fig.add_subplot(111, projection='3d')
    
    # Create button axes with labels
    prev_ax = fig.add_axes([0.3, 0.02, 0.1, 0.05])
    next_ax = fig.add_axes([0.6, 0.02, 0.1, 0.05])
    prev_ax.set_label('prev_button')
    next_ax.set_label('next_button')
    
    # Create buttons
    prev_button = Button(prev_ax, 'Previous')
    next_button = Button(next_ax, 'Next')

    def update_plot():
        # Clear the text from previous event number
        for txt in fig.texts:
            txt.remove()
            
        # Update the plot
        plot_event(event_number, tree, metadata, main_ax, fig)
        
        # Add event counter
        fig.text(0.5, 0.02, f'Event {event_number}/{n_events-1}', 
                ha='center', va='center')
        
        # Redraw the canvas
        fig.canvas.draw_idle()

    def on_prev(event):
        nonlocal event_number
        event_number = (event_number - 1) % n_events
        update_plot()

    def on_next(event):
        nonlocal event_number
        event_number = (event_number + 1) % n_events
        update_plot()

    def on_key(event):
        nonlocal event_number
        if event.key == 'left':
            event_number = (event_number - 1) % n_events
            update_plot()
        elif event.key == 'right':
            event_number = (event_number + 1) % n_events
            update_plot()
        elif event.key == 'q':
            plt.close(fig)

    # Connect callbacks
    prev_button.on_clicked(on_prev)
    next_button.on_clicked(on_next)
    fig.canvas.mpl_connect('key_press_event', on_key)

    # Initial plot
    update_plot()

    print("\nNavigation:")
    print("- Use 'Previous' and 'Next' buttons")
    print("- Or use Left/Right arrow keys")
    print("- Press 'q' to quit")
    print("- Click and drag to rotate the 3D view")

    plt.show(block=True)

if __name__ == "__main__":
    main() 