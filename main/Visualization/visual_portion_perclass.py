import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines  # For creating custom legend entries

f1_per_class = np.array([np.array([[[0.        , 0.        , 0.59976663, 0.        , 0.        ],
        [0.        , 0.        , 0.66666667, 0.        , 0.        ],
        [0.        , 0.        , 0.6962552 , 0.        , 0.        ],
        [0.        , 0.        , 0.65131579, 0.        , 0.        ],
        [0.        , 0.        , 0.7020649 , 0.        , 0.        ],
        [0.        , 0.        , 0.50641774, 0.        , 0.        ],
        [0.        , 0.        , 0.55721393, 0.        , 0.        ],
        [0.        , 0.        , 0.62068966, 0.        , 0.        ]],

       [[0.30625   , 0.        , 0.63583062, 0.15942029, 0.        ],
        [0.25874126, 0.        , 0.69710983, 0.09803922, 0.        ],
        [0.30449827, 0.        , 0.73267327, 0.25      , 0.        ],
        [0.38728324, 0.        , 0.68951194, 0.09876543, 0.        ],
        [0.22551929, 0.        , 0.73795319, 0.1375    , 0.        ],
        [0.18897638, 0.        , 0.5136541 , 0.09090909, 0.        ],
        [0.35547576, 0.        , 0.54353133, 0.02739726, 0.        ],
        [0.24633431, 0.        , 0.66511628, 0.23255814, 0.        ]]]), np.array([[[0.        , 0.        , 0.59976663, 0.        , 0.        ],
        [0.        , 0.        , 0.66666667, 0.        , 0.        ],
        [0.        , 0.        , 0.6962552 , 0.        , 0.        ],
        [0.        , 0.        , 0.65131579, 0.        , 0.        ],
        [0.        , 0.        , 0.7020649 , 0.        , 0.        ],
        [0.        , 0.        , 0.50641774, 0.        , 0.        ],
        [0.        , 0.        , 0.55721393, 0.        , 0.        ],
        [0.        , 0.        , 0.62068966, 0.        , 0.        ]],

       [[0.66058394, 0.        , 0.7357513 , 0.        , 0.        ],
        [0.68725869, 0.        , 0.78862314, 0.        , 0.        ],
        [0.47179487, 0.        , 0.69145729, 0.        , 0.        ],
        [0.54117647, 0.        , 0.67058824, 0.        , 0.        ],
        [0.54644809, 0.        , 0.80614774, 0.        , 0.        ],
        [0.80140187, 0.        , 0.65483871, 0.        , 0.        ],
        [0.61251664, 0.        , 0.74012856, 0.        , 0.        ],
        [0.65195246, 0.        , 0.70349908, 0.        , 0.        ]]]), np.array([[[0.82134571, 0.06944444, 0.85596708, 0.81111111, 0.74888559],
        [0.77241379, 0.05714286, 0.9048414 , 0.81188119, 0.62697023],
        [0.94035088, 0.        , 0.89218595, 0.79508197, 0.80733945],
        [0.9787234 , 0.        , 0.76942675, 0.69158879, 0.66666667],
        [0.88220551, 0.03252033, 0.92307692, 0.93449782, 0.7699115 ],
        [0.89506953, 0.05194805, 0.87115165, 0.69767442, 0.85536547],
        [0.91750503, 0.07017544, 0.88912134, 0.90402477, 0.79671458],
        [0.90640394, 0.08510638, 0.86825054, 0.80291971, 0.8372093 ]],

       [[0.88118812, 0.17834395, 0.84200196, 0.63448276, 0.77925926],
        [0.83      , 0.24752475, 0.9104355 , 0.73913043, 0.68189807],
        [0.92473118, 0.09090909, 0.85894737, 0.83794466, 0.75      ],
        [0.98591549, 0.        , 0.80325644, 0.6873065 , 0.8705036 ],
        [0.84383562, 0.18571429, 0.92261185, 0.92093023, 0.77620397],
        [0.90556274, 0.26      , 0.91055046, 0.74025974, 0.89863843],
        [0.92537313, 0.09230769, 0.88586387, 0.82352941, 0.8302583 ],
        [0.9009434 , 0.19607843, 0.85365854, 0.80176211, 0.83615819]]]), np.array([[[0.87128713, 0.35576923, 0.89303238, 0.79761905, 0.8452579 ],
        [0.81795511, 0.48763251, 0.92610837, 0.80769231, 0.78059072],
        [0.88815789, 0.31578947, 0.82480958, 0.82644628, 0.75331565],
        [0.98591549, 0.10526316, 0.84731774, 0.71197411, 0.93023256],
        [0.905     , 0.42528736, 0.93126935, 0.93636364, 0.83159463],
        [0.88997555, 0.33962264, 0.93224299, 0.75342466, 0.85025818],
        [0.91269841, 0.35135135, 0.87690743, 0.8807947 , 0.80962801],
        [0.90398126, 0.3255814 , 0.85894737, 0.83544304, 0.81764706]],

       [[0.88944724, 0.42790698, 0.90731707, 0.84615385, 0.87931034],
        [0.83248731, 0.54609929, 0.9253012 , 0.80392157, 0.77899344],
        [0.94699647, 0.5625    , 0.91881188, 0.82730924, 0.91503268],
        [0.98591549, 0.43478261, 0.82900137, 0.69453376, 0.9347079 ],
        [0.8976378 , 0.32258065, 0.93398533, 0.92857143, 0.82748538],
        [0.88805031, 0.26923077, 0.91799544, 0.75342466, 0.84444444],
        [0.92900609, 0.41025641, 0.88306452, 0.87707641, 0.84210526],
        [0.92944039, 0.28947368, 0.86409736, 0.80869565, 0.82492582]]])])

categories = ['W', 'N1', 'N2', 'N3', 'REM']


# Function to create vertical dumbbell plots with subplots
def vertical_dumbbell_plot_subplots():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # Create a 2x2 grid of subplots
    axes = axes.flatten()  # Flatten the 2D array of axes for easier iteration

    for i, ax in enumerate(axes):  # Iterate over each subplot and each condition
        for j in range(len(categories)):
            # Plot the dumbbell even if both original and augmented values are 0
            if f1_per_class[0][i][j] == 0 and f1_per_class[1][i][j] == 0:
                # Show zero points using a distinct color
                ax.plot([j, j], [0, 0], color='grey', lw=2)
                ax.scatter(j, 0, color='red', s=100, label='Original & Augmented Zero' if j == 0 else "")
            else:
                # Plot normal dumbbell plots for non-zero values
                ax.plot([j, j], [f1_per_class[0][i][j], f1_per_class[1][i][j]], color='grey', lw=2)
                ax.scatter(j, f1_per_class[0][i][j], color='blue', s=100, label='Original' if j == 0 else "")
                ax.scatter(j, f1_per_class[1][i][j], color='green', s=100, label='Augmented' if j == 0 else "")

        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)  # Set limits for the F1 score axis (0 to 1)
        ax.set_title(f'Condition {i + 1}')  # Set title for each subplot
        ax.set_ylabel('F1 Score')

    # Custom legend entry for the red zero dots
    red_dot = mlines.Line2D([], [], color='red', marker='o', linestyle='None', markersize=10,
                            label='Original & Augmented Zero')
    blue_dot = mlines.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=10, label='Original')
    green_dot = mlines.Line2D([], [], color='green', marker='o', linestyle='None', markersize=10, label='Augmented')

    # Add custom legend to the plot
    fig.legend(handles=[blue_dot, green_dot, red_dot], loc='upper center', ncol=3)

    fig.suptitle('Vertical Dumbbell Plots for F1 Per Class Across 4 Conditions')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make space for the main title
    plt.savefig('Vertical_Dumbbell_Plot_Subplots_With_Legend.svg')
    plt.show()


vertical_dumbbell_plot_subplots()