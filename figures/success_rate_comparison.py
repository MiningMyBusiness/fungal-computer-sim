import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(10, 8))

categories = ['Random placement', 'Current invention']
success_rates = [0.002, 89.0]
colors = ['#d62728', '#2ca02c']

bars = ax.bar(categories, success_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

ax.set_ylabel('Success Rate (%)', fontsize=14, fontweight='bold')
ax.set_title('Success rate comparison with Ping-and-Predict vs. random electrode/stimulus placement', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_ylim(0, 100)

for i, (bar, rate) in enumerate(zip(bars, success_rates)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{rate}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.text(bar.get_x() + bar.get_width()/2., height/2,
            'N=5000',
            ha='center', va='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.8))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3, linestyle='--')

caption_text = ("We simulated 10,000 random fungal specimens with the Digital Twin Simulator with electrical "
                "parameters within the biological range. We carried out the Ping-and-Predict workflow on 5,000 "
                "and on the other 5,000, we placed electrodes and applied stimuli randomly. The plot shows the "
                "success rate of creating a performant XOR gate.")

fig.text(0.5, 0.02, caption_text, ha='center', fontsize=10, style='italic', wrap=True)

plt.tight_layout(rect=[0, 0.08, 1, 1])

plt.savefig('/Users/kiranbhattacharyya/Documents/fungal-computer-sim/figures/success_rate_comparison.png', 
            dpi=300, bbox_inches='tight')
plt.savefig('/Users/kiranbhattacharyya/Documents/fungal-computer-sim/figures/success_rate_comparison.pdf', 
            bbox_inches='tight')

plt.show()

print("Success rate comparison plot saved as success_rate_comparison.png and success_rate_comparison.pdf")
