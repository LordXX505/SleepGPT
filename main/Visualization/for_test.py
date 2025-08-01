import matplotlib.pyplot as plt

# Data for the bar chart
categories = ['Direct Costs', 'Indirect Costs']
costs = [51.2, 36.6]  # Maximum values in billion USD

# Create the bar chart
plt.figure(figsize=(6, 4))
plt.bar(categories, costs, color=['blue', 'orange'])

# Adding labels and title
plt.ylabel('Cost (Billion USD)')
plt.title('Maximum Estimated Costs of Insomnia')
plt.savefig('./result/cost.svg')
# Display the plot
plt.show()