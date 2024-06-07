import os
import requests
import matplotlib.pyplot as plt

TELEGRAM_BOT_TOKEN = os.environ['TELEGRAM_BOT_TOKEN']
TELEGRAM_CHAT_ID = os.environ['TELEGRAM_CHAT_ID']

# The tool configuration
tool_config = {
    "type": "function",
    "function": {
        "name": "create_energy_consumption_chart",
        "description": "Create a pie chart illustrating the user's energy consumption across three categories and send it via Telegram.",
        "parameters": {
            "type": "object",
            "properties": {
                "food": {
                    "type": "object",
                    "description": "Energy consumption for food.",
                    "properties": {
                        "kWh": {"type": "number", "description": "Energy in kWh."},
                        "CO2": {"type": "number", "description": "CO2 emissions in kg."}
                    },
                    "required": ["kWh", "CO2"]
                },
                "home_energy_use": {
                    "type": "object",
                    "description": "Energy consumption for home energy use.",
                    "properties": {
                        "kWh": {"type": "number", "description": "Energy in kWh."},
                        "CO2": {"type": "number", "description": "CO2 emissions in kg."}
                    },
                    "required": ["kWh", "CO2"]
                },
                "transport": {
                    "type": "object",
                    "description": "Energy consumption for transport.",
                    "properties": {
                        "kWh": {"type": "number", "description": "Energy in kWh."},
                        "CO2": {"type": "number", "description": "CO2 emissions in kg."}
                    },
                    "required": ["kWh", "CO2"]
                }
            },
            "required": ["food", "home_energy_use", "transport"]
        }
    }
}

# The callback function (Creates pie chart and sends it via Telegram)
def create_energy_consumption_chart(arguments):
    """
    Create a pie chart illustrating the user's energy consumption across three categories and send it via Telegram.

    :param arguments: dict, Contains the necessary information for creating the chart.
                      Expected keys: food, home_energy_use, transport.
    :return: dict or str, Response from the Telegram API or error message.
    """
    # Extracting information from arguments
    food = arguments.get('food')
    home_energy_use = arguments.get('home_energy_use')
    transport = arguments.get('transport')

    # Validating the presence of all required information
    if not all([food, home_energy_use, transport]):
        return "Missing required information. Please provide energy consumption for food, home energy use, and transport."

    # Data for the pie chart
    categories = ['Food', 'Home Energy Use', 'Transport']
    energy_kwh = [food['kWh'], home_energy_use['kWh'], transport['kWh']]
    co2_kg = [food['CO2'], home_energy_use['CO2'], transport['CO2']]

    # Plotting the pie chart
    fig, ax = plt.subplots()
    ax.pie(energy_kwh, labels=categories, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    plt.title('Energy Consumption Breakdown')

    # Adding annotations for CO2 emissions
    for i, (kw, co2) in enumerate(zip(energy_kwh, co2_kg)):
        plt.annotate(f'{kw} kWh\n{co2} kg CO2', xy=(0.7, 0.5 - i * 0.1), color='black', fontsize=10)

    # Save the pie chart as an image
    chart_path = '/tmp/energy_consumption_chart.png'
    plt.savefig(chart_path)
    plt.close(fig)

    # Send the pie chart via Telegram
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    files = {'photo': open(chart_path, 'rb')}
    data = {'chat_id': TELEGRAM_CHAT_ID}
    response = requests.post(url, files=files, data=data)
    if response.status_code == 200:
        return {"status": "success", "message": "Pie chart sent successfully via Telegram."}
    else:
        return {"status": "error", "message": f"Failed to send pie chart: {response.text}"}
