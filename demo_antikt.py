import uproot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arrow, Wedge
from mpl_toolkits.mplot3d import Axes3D
import argparse
import sys
from matplotlib.widgets import Button
import matplotlib.colors
import matplotlib.animation as animation

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
    
    # Sort jets by pT
    jet_indices = np.argsort(jet_pt)[::-1]
    
    # Find leading particles in each jet
    leading_particles = {}  # Dictionary to store leading particle index for each jet
    leading_pt = {}        # Dictionary to store leading particle pT
    for jetidx in range(len(jet_pt)):
        jet_particles = [(i, pt) for i, (pt, idx) in enumerate(zip(particle_pt, particle_jetIndex)) if idx == jetidx]
        if jet_particles:
            max_particle = max(jet_particles, key=lambda x: x[1])
            leading_particles[jetidx] = max_particle[0]
            leading_pt[jetidx] = max_particle[1]
    
    # First plot all particles
    for i, (eta, phi, pt, jetidx) in enumerate(zip(particle_eta, particle_phi, particle_pt, particle_jetIndex)):
        if jetidx >= 0:
            # Check if this is the leading particle in its jet
            if i == leading_particles.get(jetidx, -1):
                color = 'darkred'  # Same color for all leading particles
                alpha = 1.0
                linewidth = 3.5  # Thicker line for leading particles
                linestyle = ':'  # Dotted line for leading particles
            else:
                color = jet_colors[jetidx % 9]
                alpha = 0.6
                linewidth = 1.5
                linestyle = '-'  # Solid line for other particles
        else:
            color = 'lightgray'
            alpha = 0.2
            linewidth = 1.5
            linestyle = '-'
            
        ax.plot([eta, eta], [phi, phi], [0, pt], 
                color=color,
                alpha=alpha,
                linewidth=linewidth,
                linestyle=linestyle)
    
    # Plot jets
    R = 0.4  # Anti-kT parameter
    for idx, i in enumerate(jet_indices):
        eta, phi, pt, area = jet_eta[i], jet_phi[i], jet_pt[i], jet_area[i]
        color = jet_colors[i % 9]
        
        # Draw jet cone for leading and subleading jets
        if idx < 2:  # Leading and subleading jets
            z_points = np.linspace(0, pt, 30)
            for t in np.linspace(0, 2*np.pi, 20):
                r = R * (1 - z_points/pt)  # Reverse cone: wider at top
                cone_eta = eta + r * np.cos(t)
                cone_phi = phi + r * np.sin(t)
                
                ax.plot(cone_eta, cone_phi, z_points,
                       color=color,
                       alpha=0.05,  # More transparent
                       linewidth=1)
        
        # Draw jet axis
        ax.plot([eta, eta], [phi, phi], [0, pt], 
                color=color,
                linestyle='-',
                linewidth=2,
                alpha=1.0)
        
        # Draw jet radius circle at z=0
        theta = np.linspace(0, 2*np.pi, 100)
        circle_eta = eta + R * np.cos(theta)
        circle_phi = phi + R * np.sin(theta)
        circle_z = np.zeros_like(theta)
        ax.plot(circle_eta, circle_phi, circle_z, 
                color=color, 
                linestyle='-', 
                alpha=0.8)
        
        # Add jet marker at the top
        ax.scatter([eta], [phi], [pt], 
                  color=color,
                  s=80,
                  marker='^',
                  label=f'Jet {i}: pT={pt:.1f} GeV')

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
    ax.set_zlim(0, max_pt * 1.2)
    
    # Set initial view angle for better visualization
    ax.view_init(elev=25, azim=45)
    
    # Add grid for better depth perception
    ax.grid(True, alpha=0.3)

    # Update legend
    if len(jet_pt) > 0:
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='lightgray', alpha=0.2, linewidth=1.5, label='Unassociated particles'),
            Line2D([0], [0], color='darkred', linewidth=3.5, linestyle=':', 
                  label=f'Leading particles (pT={max(leading_pt.values()):.1f} GeV)'),
            Line2D([0], [0], marker='^', color='red', markersize=10, label=f'Jets (R={R})')
        ]
        # Add specific jet information
        for i, pt in enumerate(jet_pt[jet_indices]):
            jet_idx = jet_indices[i]
            legend_elements.append(
                Line2D([0], [0], color=jet_colors[jet_idx % 9],
                      linewidth=1.5,
                      label=f'Jet {jet_idx}: pT={pt:.1f} GeV')
            )
        ax.legend(handles=legend_elements, loc='upper right')

    return fig, ax

def demonstrate_antikt(event_number, tree):
    """Create a step-by-step demonstration of the anti-kt algorithm"""
    # Create figure with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Anti-kT Algorithm Demonstration', fontsize=16)
    
    # Get data for the specific event
    particle_pt = tree['particle_pt'].array()[event_number]
    particle_eta = tree['particle_eta'].array()[event_number]
    particle_phi = tree['particle_phi'].array()[event_number]
    
    # Step 1: Show initial particles
    ax1 = axs[0,0]
    ax1.set_title('Step 1: Initial Particles')
    scatter = ax1.scatter(particle_eta, particle_phi, 
                         s=particle_pt*10, # size proportional to pT
                         alpha=0.6,
                         c=particle_pt,    # color by pT
                         cmap='viridis')
    fig.colorbar(scatter, ax=ax1, label='pT [GeV]')
    
    # Add explanation text for Step 1
    ax1.text(-3.8, 2.8, 
             'Input particles with:\n' + 
             r'$p_T$ = transverse momentum' + '\n' +
             r'$\eta$ = pseudorapidity' + '\n' +
             r'$\phi$ = azimuthal angle',
             verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.7))
    
    ax1.set_xlabel('η')
    ax1.set_ylabel('φ')
    ax1.grid(True, alpha=0.3)
    
    # Step 2: Distance Calculation
    ax2 = axs[0,1]
    ax2.set_title('Step 2: Multiple Jets Formation')
    
    # Show particles
    ax2.scatter(particle_eta, particle_phi, 
                s=particle_pt*10,
                alpha=0.6,
                c=particle_pt,
                cmap='viridis')
    
    # Find the three highest pT particles that are far apart (potential jet seeds)
    jet_seeds = []
    pt_sorted_indices = np.argsort(particle_pt)[::-1]
    
    for idx in pt_sorted_indices:
        # Check if this particle is far enough from existing seeds
        is_isolated = True
        for seed_idx in jet_seeds:
            dR = np.sqrt((particle_eta[idx]-particle_eta[seed_idx])**2 + 
                        (particle_phi[idx]-particle_phi[seed_idx])**2)
            if dR < 0.8:  # 2*R to ensure well-separated jets
                is_isolated = False
                break
        if is_isolated:
            jet_seeds.append(idx)
            if len(jet_seeds) == 3:  # Stop after finding 3 seeds
                break
    
    # Colors for different jets
    jet_colors = ['red', 'blue', 'green']
    
    # For each jet seed, show closest particles
    for seed_idx, base_color in zip(jet_seeds, jet_colors):
        # Calculate distances to all other particles
        distances = []
        for i in range(len(particle_pt)):
            if i != seed_idx:
                dR = np.sqrt((particle_eta[seed_idx]-particle_eta[i])**2 + 
                            (particle_phi[seed_idx]-particle_phi[i])**2)
                dij = min(1/particle_pt[seed_idx]**2, 1/particle_pt[i]**2) * dR**2/0.4**2
                distances.append((i, dij, dR))
        
        # Sort by dij
        distances.sort(key=lambda x: x[1])
        
        # Show 5 closest particles for each seed
        colors = [matplotlib.colors.to_rgba(base_color, alpha=1.0 - i*0.15) for i in range(5)]
        
        for (idx, dij, dR), color in zip(distances[:5], colors):
            ax2.plot([particle_eta[seed_idx], particle_eta[idx]],
                    [particle_phi[seed_idx], particle_phi[idx]],
                    c=color, linewidth=2, alpha=0.8, 
                    label=f'dij = {dij:.3e} ({base_color} jet)')
            
            ax2.annotate(f'{particle_pt[idx]:.1f} GeV',
                        (particle_eta[idx], particle_phi[idx]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8,
                        color=color)
        
        # Annotate jet seed
        ax2.annotate(f'{particle_pt[seed_idx]:.1f} GeV',
                    (particle_eta[seed_idx], particle_phi[seed_idx]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8,
                    color=base_color,
                    weight='bold')
        
        # Draw jet cone
        circle = Circle((particle_eta[seed_idx], particle_phi[seed_idx]),
                       radius=0.4,  # R parameter
                       fill=False,
                       color=base_color,
                       linestyle='--',
                       alpha=0.5)
        ax2.add_patch(circle)
    
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_xlabel('η')
    ax2.set_ylabel('φ')
    ax2.grid(True, alpha=0.3)
    
    # Step 3: Initial Clustering
    ax3 = axs[1,0]
    ax3.set_title('Step 3: Clustering Process')
    
    # Show all particles in background
    ax3.scatter(particle_eta, particle_phi,
                s=particle_pt*10,
                alpha=0.2,  # More transparent
                c='gray')
    
    # Show clustering for the three jets
    for seed_idx, color in zip(jet_seeds, ['red', 'blue', 'green']):
        # Get closest particles for this seed
        distances = []
        for i in range(len(particle_pt)):
            if i != seed_idx and i not in jet_seeds:  # Exclude other seeds
                dR = np.sqrt((particle_eta[seed_idx]-particle_eta[i])**2 + 
                            (particle_phi[seed_idx]-particle_phi[i])**2)
                if dR < 0.4:  # Only consider particles within jet radius
                    dij = min(1/particle_pt[seed_idx]**2, 1/particle_pt[i]**2) * dR**2/0.4**2
                    distances.append((i, dij, dR))
        
        # Sort by dij and get closest 3 particles
        distances.sort(key=lambda x: x[1])
        closest_particles = distances[:3]
        
        # Draw arrows to show clustering sequence
        for i, (idx, dij, dR) in enumerate(closest_particles):
            # Draw arrow from seed to particle
            ax3.arrow(particle_eta[seed_idx], particle_phi[seed_idx],
                     particle_eta[idx] - particle_eta[seed_idx],
                     particle_phi[idx] - particle_phi[seed_idx],
                     head_width=0.1, head_length=0.1, fc=color, ec=color,
                     alpha=0.8-i*0.2,  # Fade out later steps
                     length_includes_head=True)
            
            # Show clustering order
            ax3.annotate(f'{i+1}',
                        (particle_eta[idx], particle_phi[idx]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, color=color, weight='bold')
        
        # Draw jet cone
        circle = Circle((particle_eta[seed_idx], particle_phi[seed_idx]),
                       radius=0.4,
                       fill=False,
                       color=color,
                       linestyle='--',
                       alpha=0.5)
        ax3.add_patch(circle)
        
        # Label seed
        ax3.annotate(f'Seed\n{particle_pt[seed_idx]:.1f} GeV',
                    (particle_eta[seed_idx], particle_phi[seed_idx]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, color=color, weight='bold')
    
    # Add explanation text
    ax3.text(-3.8, 2.8,
             'Clustering sequence:\n' +
             '1. Start from highest pT seeds\n' +
             '2. Merge closest particles first\n' +
             '3. Update jet momentum after each merge:\n' +
             r'$p_T^{new} = p_{T,i} + p_{T,j}$' + '\n' +
             r'$\eta^{new} = \frac{p_{T,i}\eta_i + p_{T,j}\eta_j}{p_T^{new}}$' + '\n' +
             r'$\phi^{new} = \frac{p_{T,i}\phi_i + p_{T,j}\phi_j}{p_T^{new}}$',
             verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.7))
    
    ax3.legend(['Background particles', 'First jet', 'Second jet', 'Third jet'],
               loc='upper right')
    ax3.set_xlabel('η')
    ax3.set_ylabel('φ')
    ax3.grid(True, alpha=0.3)
    
    # Step 4: Final Jets
    ax4 = axs[1,1]
    ax4.set_title('Step 4: Final Jets')
    
    # Get jet data
    jet_pt = tree['jet_pt'].array()[event_number]
    jet_eta = tree['jet_eta'].array()[event_number]
    jet_phi = tree['jet_phi'].array()[event_number]
    particle_jetIndex = tree['particle_jetIndex'].array()[event_number]
    
    # Show constituent particles first
    colors = plt.cm.Set1(np.linspace(0, 1, len(jet_pt)))
    for i, (eta, phi, pt, jetidx) in enumerate(zip(particle_eta, particle_phi, 
                                                  particle_pt, particle_jetIndex)):
        if jetidx >= 0:  # Particle belongs to a jet
            ax4.scatter(eta, phi, 
                       s=pt*5,
                       color=colors[jetidx],
                       alpha=0.3)
    
    # Show final jets
    jet_scatter = ax4.scatter(jet_eta, jet_phi,
                             s=jet_pt*20,
                             c=jet_pt,
                             cmap='viridis',
                             marker='^',
                             label='Jets')
    
    # Add combined explanation text for Step 4 and algorithm details
    ax4.text(0.5, -2.8,
             'Anti-kT Algorithm Summary:\n\n' +
             'Distance metrics:\n' + 
             r'$d_{ij} = \min(p_{T,i}^{-2}, p_{T,j}^{-2})\frac{\Delta R_{ij}^2}{R^2}$' + '\n' +
             r'$\Delta R_{ij}^2 = (\eta_i - \eta_j)^2 + (\phi_i - \phi_j)^2$' + '\n' +
             r'$d_{iB} = p_{T,i}^{-2}$' + ' (beam distance)\n\n' +
             'Algorithm steps:\n' +
             '1. Find minimum of all dij and diB\n' +
             '2. If dij is min: merge i & j\n' +
             '3. If diB is min: jet is complete\n' +
             '4. Repeat until all particles assigned\n\n',
             verticalalignment='bottom',
             horizontalalignment='left',
             bbox=dict(facecolor='white', alpha=0.7))
    
    fig.colorbar(jet_scatter, ax=ax4, label='Jet pT [GeV]')
    ax4.legend(loc='upper right')
    ax4.set_xlabel('η')
    ax4.set_ylabel('φ')
    ax4.grid(True, alpha=0.3)
    
    # Set consistent axis limits for all plots
    for ax in axs.flat:
        ax.set_xlim(-4, 4)
        ax.set_ylim(-np.pi, np.pi)
    
    plt.tight_layout()
    return fig

def animate_antikt(event_number, tree):
    """Create an animated demonstration of the anti-kt algorithm"""
    import matplotlib.animation as animation
    from matplotlib.patches import Circle, Wedge
    
    # Create figure with four areas
    fig = plt.figure(figsize=(15, 12))
    gs = plt.GridSpec(3, 2, height_ratios=[3, 1, 1])
    ax_main = fig.add_subplot(gs[0, :])
    ax_explain = fig.add_subplot(gs[1, 0])
    ax_metrics = fig.add_subplot(gs[1, 1])
    ax_text = fig.add_subplot(gs[2, :])
    
    fig.suptitle('Anti-kT Algorithm Demonstration', fontsize=14, y=0.98)
    
    # Get data for the specific event
    particle_pt = tree['particle_pt'].array()[event_number]
    particle_eta = tree['particle_eta'].array()[event_number]
    particle_phi = tree['particle_phi'].array()[event_number]
    
    # Take only top 20 pT particles
    top_n = 20
    indices = np.argsort(particle_pt)[::-1][:top_n]
    particle_pt = particle_pt[indices]
    particle_eta = particle_eta[indices]
    particle_phi = particle_phi[indices]
    
    # Define number of particles after selection
    n_particles = len(particle_pt)
    
    # Initialize clustering state
    particle_status = np.zeros(n_particles)  # 0: free, 1: clustered, 2: beam
    clustering_sequence = []
    current_jet_id = 0
    clustered_particles = {}  # jet_id -> list of particle indices
    jet_centers = {}  # jet_id -> (eta, phi, pt)
    
    def calculate_dij():
        all_dij = []
        for i in range(n_particles):
            if particle_status[i] == 0:
                # Beam distance
                diB = 1/particle_pt[i]**2
                all_dij.append((i, 'B', diB, 0.0))  # dR=0 for beam
                
                # Particle distances
                for j in range(i+1, n_particles):
                    if particle_status[j] == 0:
                        dR = np.sqrt((particle_eta[i]-particle_eta[j])**2 + 
                                   (particle_phi[i]-particle_phi[j])**2)
                        kt = min(1/particle_pt[i]**2, 1/particle_pt[j]**2)
                        dij = kt * dR**2/0.4**2
                        all_dij.append((i, j, dij, dR))
        
        return sorted(all_dij, key=lambda x: x[2])
    
    # Generate full clustering sequence
    while True:
        dij_list = calculate_dij()
        if not dij_list:
            break
            
        clustering_sequence.append(dij_list[:3])  # Store top 3 candidates
        i, j, dij, dR = dij_list[0]  # Get minimum distance
        
        if j == 'B':
            particle_status[i] = 2
        else:
            if current_jet_id not in clustered_particles:
                # New jet
                clustered_particles[current_jet_id] = [i, j]
                particle_status[i] = particle_status[j] = 1
                total_pt = particle_pt[i] + particle_pt[j]
                jet_centers[current_jet_id] = (
                    (particle_eta[i] * particle_pt[i] + particle_eta[j] * particle_pt[j]) / total_pt,
                    (particle_phi[i] * particle_pt[i] + particle_phi[j] * particle_pt[j]) / total_pt,
                    total_pt
                )
                current_jet_id += 1
            else:
                # Add to existing jet
                clustered_particles[current_jet_id].append(i)
                particle_status[i] = 1
                jet_eta, jet_phi, jet_pt = jet_centers[current_jet_id]
                total_pt = jet_pt + particle_pt[i]
                jet_centers[current_jet_id] = (
                    (jet_eta * jet_pt + particle_eta[i] * particle_pt[i]) / total_pt,
                    (jet_phi * jet_pt + particle_phi[i] * particle_pt[i]) / total_pt,
                    total_pt
                )
    
    # Initialize state for animation
    particle_states = []  # Will store state at each step
    jet_states = []      # Will store jet info at each step
    
    # Initial state
    current_particles = {
        'status': np.zeros(n_particles),  # 0: free, 1: clustered, 2: beam
        'jets': {},      # jet_id -> list of particle indices
        'centers': {}    # jet_id -> (eta, phi, pt)
    }
    particle_states.append(current_particles.copy())
    jet_states.append({})
    
    # Generate states for each step
    current_jet_id = 0
    for step in range(len(clustering_sequence)):
        i, j, dij, dR = clustering_sequence[step][0]
        
        # Deep copy current state
        new_state = {
            'status': current_particles['status'].copy(),
            'jets': {k: v.copy() for k, v in current_particles['jets'].items()},
            'centers': {k: v for k, v in current_particles['centers'].items()}
        }
        
        if j == 'B':
            new_state['status'][i] = 2
        else:
            if current_jet_id not in new_state['jets']:
                # New jet
                new_state['jets'][current_jet_id] = [i, j]
                new_state['status'][i] = new_state['status'][j] = 1
                total_pt = particle_pt[i] + particle_pt[j]
                new_state['centers'][current_jet_id] = (
                    (particle_eta[i] * particle_pt[i] + particle_eta[j] * particle_pt[j]) / total_pt,
                    (particle_phi[i] * particle_pt[i] + particle_phi[j] * particle_pt[j]) / total_pt,
                    total_pt
                )
                current_jet_id += 1
            else:
                # Add to existing jet
                new_state['jets'][current_jet_id].append(i)
                new_state['status'][i] = 1
                jet_eta, jet_phi, jet_pt = new_state['centers'][current_jet_id]
                total_pt = jet_pt + particle_pt[i]
                new_state['centers'][current_jet_id] = (
                    (jet_eta * jet_pt + particle_eta[i] * particle_pt[i]) / total_pt,
                    (jet_phi * jet_pt + particle_phi[i] * particle_pt[i]) / total_pt,
                    total_pt
                )
        
        particle_states.append(new_state)
        jet_states.append(new_state['centers'])
        current_particles = new_state
    
    def update(frame):
        # Clear the entire figure first
        plt.clf()
        
        # Recreate the GridSpec and subplots for each frame
        gs = plt.GridSpec(3, 2, height_ratios=[3, 1, 1])
        ax_main = fig.add_subplot(gs[0, :])
        ax_explain = fig.add_subplot(gs[1, 0])
        ax_metrics = fig.add_subplot(gs[1, 1])
        ax_text = fig.add_subplot(gs[2, :])
        
        # Set the main title
        fig.suptitle('Anti-kT Algorithm Demonstration', fontsize=14, y=0.98)
        
        # Remove boxes from all subplots
        for ax in [ax_main, ax_explain, ax_metrics, ax_text]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        # Set up main plot
        ax_main.set_xlim(-4, 4)
        ax_main.set_ylim(-np.pi, np.pi)
        ax_main.set_xlabel('η')
        ax_main.set_ylabel('φ')
        ax_main.grid(True, alpha=0.3)
        ax_main.set_title('Particle Clustering', pad=20)
        
        # Get current state
        state = particle_states[min(frame, len(particle_states)-1)]
        
        # Show unclustered particles in main plot
        unclustered = state['status'] == 0
        ax_main.scatter(particle_eta[unclustered], particle_phi[unclustered],
                       s=particle_pt[unclustered]*10,
                       c='gray',
                       alpha=0.6,
                       label='Unclustered')
        
        # Show jets and their particles in main plot
        colors = plt.cm.Set1(np.linspace(0, 1, max(1, len(state['jets']))))
        for jet_id, particles in state['jets'].items():
            color = colors[jet_id % len(colors)]
            jet_eta, jet_phi, jet_pt = state['centers'][jet_id]
            
            # Draw jet cone
            circle = Circle((jet_eta, jet_phi),
                          radius=0.4,
                          fill=False,
                          color=color,
                          linestyle='--',
                          alpha=0.5)
            ax_main.add_patch(circle)
            
            # Show jet center and particles
            ax_main.scatter(jet_eta, jet_phi,
                          s=jet_pt*20,
                          c=[color],
                          marker='^',
                          label=f'Jet {jet_id}: pT={jet_pt:.1f} GeV')
            
            for idx in particles:
                ax_main.scatter(particle_eta[idx], particle_phi[idx],
                              s=particle_pt[idx]*10,
                              c=[color],
                              alpha=0.6)
        
        # Show current clustering step in main plot
        if frame < len(clustering_sequence):
            i, j, dij, dR = clustering_sequence[frame][0]
            if j != 'B':
                ax_main.plot([particle_eta[i], particle_eta[j]],
                           [particle_phi[i], particle_phi[j]],
                           'r--',
                           linewidth=2,
                           label='Current clustering')
        
        # Create step visualization in explanation subplot
        ax_explain.set_title('Clustering Process')
        if frame < len(clustering_sequence):
            # Show all particles
            ax_explain.scatter(particle_eta, particle_phi,
                             s=particle_pt*5,  # Smaller points
                             alpha=0.2,
                             c='gray')
            
            # Show current clustering step
            i, j, dij, dR = clustering_sequence[frame][0]
            if j != 'B':
                # Draw arrow from i to j
                ax_explain.arrow(particle_eta[i], particle_phi[i],
                               particle_eta[j] - particle_eta[i],
                               particle_phi[j] - particle_phi[i],
                               head_width=0.1, head_length=0.1,
                               fc='red', ec='red',
                               alpha=0.8,
                               length_includes_head=True)
                
                # Annotate particles
                ax_explain.annotate(f'pT={particle_pt[i]:.1f}',
                                  (particle_eta[i], particle_phi[i]),
                                  xytext=(5, 5), textcoords='offset points',
                                  fontsize=8)
                ax_explain.annotate(f'pT={particle_pt[j]:.1f}',
                                  (particle_eta[j], particle_phi[j]),
                                  xytext=(5, 5), textcoords='offset points',
                                  fontsize=8)
                
                # Show dR and dij
                midpoint = ((particle_eta[i] + particle_eta[j])/2,
                          (particle_phi[i] + particle_phi[j])/2)
                ax_explain.annotate(f'ΔR={dR:.2f}\ndij={dij:.2e}',
                                  midpoint,
                                  xytext=(10, 10),
                                  textcoords='offset points',
                                  bbox=dict(facecolor='white', alpha=0.7))
        
        ax_explain.set_xlim(-4, 4)
        ax_explain.set_ylim(-np.pi, np.pi)
        ax_explain.set_xlabel('η')
        ax_explain.set_ylabel('φ')
        ax_explain.grid(True, alpha=0.3)
        
        # Create distance matrix visualization in metrics subplot
        if frame < len(clustering_sequence):
            # Calculate current distances for visualization
            dij_matrix = np.zeros((n_particles, n_particles))
            for i in range(n_particles):
                for j in range(i+1, n_particles):
                    if state['status'][i] == 0 and state['status'][j] == 0:
                        dR = np.sqrt((particle_eta[i]-particle_eta[j])**2 + 
                                   (particle_phi[i]-particle_phi[j])**2)
                        kt = min(1/particle_pt[i]**2, 1/particle_pt[j]**2)
                        dij = kt * dR**2/0.4**2
                        dij_matrix[i,j] = dij_matrix[j,i] = dij
            
            # Plot distance matrix
            im = ax_metrics.imshow(np.log10(dij_matrix + 1e-20),
                                 cmap='viridis',
                                 aspect='auto')
            
            # Add colorbar with removed box
            cbar = plt.colorbar(im, ax=ax_metrics)
            cbar.outline.set_visible(False)
            cbar.set_label('log₁₀(dᵢⱼ)', rotation=270, labelpad=15)
            
            # Set title and labels with adjusted positions
            ax_metrics.set_title('Distance Matrix', y=1.05)
            
            # Create tick labels with pT values
            active_particles = np.where(state['status'] == 0)[0]
            if len(active_particles) > 0:
                tick_labels = [f'{i}\n({particle_pt[i]:.1f} GeV)' 
                             for i in range(n_particles)]
                ax_metrics.set_xticks(range(n_particles))
                ax_metrics.set_yticks(range(n_particles))
                ax_metrics.set_xticklabels(tick_labels, rotation=45, ha='right')
                ax_metrics.set_yticklabels(tick_labels)
                
                # Add labels explaining the axes
                ax_metrics.set_xlabel('Particle Index (pT)', labelpad=10)
                ax_metrics.set_ylabel('Particle Index (pT)', labelpad=10)
            
            # Remove box around plot
            for spine in ax_metrics.spines.values():
                spine.set_visible(False)
            
            # Highlight current clustering pair
            i, j, dij, dR = clustering_sequence[frame][0]
            if j != 'B':
                ax_metrics.plot(j, i, 'rx', markersize=10, label='Current pair')
                ax_metrics.plot(i, j, 'rx', markersize=10)
                
                # Add annotation for current pair
                ax_metrics.annotate(f'dij={dij:.2e}',
                                  xy=(j, i),
                                  xytext=(10, 10),
                                  textcoords='offset points',
                                  bbox=dict(facecolor='white', alpha=0.7))
        else:
            ax_metrics.text(0.5, 0.5, 'Clustering Complete',
                          ha='center', va='center',
                          transform=ax_metrics.transAxes,
                          fontsize=12)
            ax_metrics.axis('off')
        
        # Show current step explanation
        if frame < len(clustering_sequence):
            i, j, dij, dR = clustering_sequence[frame][0]
            step_text = "Current Step Explanation:\n\n"
            
            if j == 'B':
                step_text += (
                    f"• Particle {i} (pT={particle_pt[i]:.1f} GeV)\n"
                    f"• Beam distance diB={dij:.2e}\n"
                    f"• Creating new jet from particle"
                )
            else:
                step_text += (
                    f"• Particles {i} and {j}\n"
                    f"• pT: {particle_pt[i]:.1f} GeV, {particle_pt[j]:.1f} GeV\n"
                    f"• Distance ΔR={dR:.2f}\n"
                    f"• Metric dij={dij:.2e}\n"
                    f"• {'Creating new jet' if current_jet_id not in state['jets'] else 'Adding to existing jet'}"
                )
        else:
            step_text = "Final Jets:\n\n"
            for jet_id, (_, _, pt) in state['centers'].items():
                n_const = len(state['jets'][jet_id])
                step_text += f"• Jet {jet_id}: pT={pt:.1f} GeV, {n_const} particles\n"
        
        ax_text.text(0.05, 0.95, step_text,
                    transform=ax_text.transAxes,
                    verticalalignment='top',
                    fontfamily='monospace',
                    bbox=dict(facecolor='white', alpha=0.9))
        ax_text.axis('off')
        
        # Show metrics with current values
        metrics_text = (
            'Distance Metrics:\n\n'
            r'$d_{ij} = \min(p_{T,i}^{-2}, p_{T,j}^{-2})\frac{\Delta R_{ij}^2}{R^2}$' '\n'
            r'$\Delta R_{ij}^2 = (\eta_i - \eta_j)^2 + (\phi_i - \phi_j)^2$' '\n'
            r'$d_{iB} = p_{T,i}^{-2}$' '\n\n'
            'Parameters:\n'
            '• R = 0.4 (jet radius)\n'
            f'• Particles: {n_particles}\n'
            f'• Current jets: {len(state["jets"])}'
        )
        ax_metrics.text(0.05, 0.01, metrics_text,
                       transform=ax_metrics.transAxes,
                       verticalalignment='top',
                       bbox=dict(facecolor='white', alpha=0.9))
        ax_metrics.axis('off')
        
        # Adjust layout
        plt.tight_layout(h_pad=2.0, w_pad=2.0)
        
        return ax_main, ax_text, ax_metrics, ax_explain
    
    # Create animation
    anim = animation.FuncAnimation(fig, update,
                                 frames=len(clustering_sequence) + 5,
                                 interval=1000,
                                 blit=True)
    
    plt.show()

def toggle_animation(event, anim):
    if anim.running:
        anim.event_source.stop()
        event.inaxes.texts[0].set_text('Play')
    else:
        anim.event_source.start()
        event.inaxes.texts[0].set_text('Pause')
    anim.running = not anim.running

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Demonstrate anti-kt algorithm')
    parser.add_argument('filename', type=str, help='Path to the ROOT file')
    parser.add_argument('-e', '--event', type=int, default=0, 
                        help='Event number to display (default: 0)')
    
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

    # Create and show the demonstration
    fig = demonstrate_antikt(event_number, tree)
    plt.show()

    # Create the animation
    animate_antikt(args.event, tree)

if __name__ == "__main__":
    main() 