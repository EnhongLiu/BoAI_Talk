import numpy as np
import matplotlib.pyplot as plt
import re

# Milkbot Prediction Function
def milk_yield_prediction_from_milkbot(parity='1', region="Europe", dim_range=np.arange(1, 306, 1)):
    parity = str(parity)

    if region == "Europe":
        if parity == '1':
            scale_p, ramp_p, decay_p, offset_p = 27.3, 30.7, 0.0028, -0.084
        elif parity == '2':
            scale_p, ramp_p, decay_p, offset_p = 34.3, 22.8, 0.00276, -0.775
        else:
            scale_p, ramp_p, decay_p, offset_p = 41.8, 26.4, 0.00359, 0.00397
    elif region == "United States":
        if parity == '1':
            scale_p, ramp_p, decay_p, offset_p = 38.3, 30.72, 0.0022, -0.499
        elif parity == '2':
            scale_p, ramp_p, decay_p, offset_p = 48.2, 22.8, 0.0029, -0.775
        else:
            scale_p, ramp_p, decay_p, offset_p = 53.5, 25.5, 0.00309, 0.0039
    else:
        raise ValueError("Invalid region specified. Choose 'Europe' or 'United States'.")

    my_prediction = []

    for dim in dim_range:
        milk_yield = scale_p * np.exp(-decay_p * dim) * (1 - np.exp((offset_p - dim) / ramp_p) / 2)
        my_prediction.append(milk_yield)

    return dim_range, my_prediction



def predict_milk_yield_for_region(region= ['Europe','United States'], parity = ['1','2','3+'], dim_range=np.arange(1,306,1)):
    
    europe_pattern = r"\b(eu|europe|eu region|european)\b"
    us_pattern = r"\b(us|united states|usa|american)\b"
    response =[]

    for r in region:
        if re.search(europe_pattern, r.lower()):
            for p in parity:
                dim_range, my_prediction = milk_yield_prediction_from_milkbot(p, region='Europe', dim_range=dim_range)
                response.append({'region': 'Europe', "parity": p, "dim_range": dim_range, "my_prediction": my_prediction})

        elif re.search(us_pattern, r.lower()):
            for p in parity:
                dim_range, my_prediction = milk_yield_prediction_from_milkbot(p, region='United States', dim_range=dim_range)
                response.append({'region': 'United States', "parity": p, "dim_range": dim_range, "my_prediction": my_prediction})

        else:
            print(f'Error: Current model only covers regions of Europe and United States. Invalid region: {r}')
    
    
    return response


# Visuals Generation Function
def get_region_colors(region, num_parities):
    region_colormap = {
        'Europe': plt.cm.Blues, # Different region have different colors
        'United States': plt.cm.Reds
    }
    
    cmap = region_colormap.get(region, plt.cm.Greys)  # Default to Greys if region not specified
    return cmap(np.linspace(0.4, 0.9, num_parities))



def milk_yield_plot(response):

    region_parity_map = {}
    
    # Collect regions and their parities
    for item in response:
        region = item['region']
        if region not in region_parity_map:
            region_parity_map[region] = []
        region_parity_map[region].append(item['parity'])

    # Generate colors for each region-parity combination
    region_colors = {}
    for region, parities in region_parity_map.items():
        unique_parities = list(set(parities))
        region_colors[region] = get_region_colors(region, len(unique_parities))

    if len(response[0]['dim_range'])>20:

        plt.figure(figsize=(8, 4))
        
        for item in response:
            region = item['region']
            parity = item['parity']
            dim_range = item['dim_range']
            milk_predictions = item['my_prediction']

            # Assign appropriate color based on region and parity
            parity_index = list(set(region_parity_map[region])).index(parity)
            plt.plot(dim_range, milk_predictions,
                     label=f'Predicted Milk - Parity {parity} in {region}',
                     color=region_colors[region][parity_index])

        plt.title('Predicted Milk Yield vs Days in Milk')
        plt.xlabel('Days in Milk')
        plt.ylabel('Milk Yield (Kg)')
        plt.legend(fontsize= 8,loc='upper right')
        plt.grid(False)
        plt.show()

    else:
        plt.figure(figsize=(8, 4))
        
        for item in response:
            region = item['region']
            parity = item['parity']
            dim_range = item['dim_range']
            milk_predictions = item['my_prediction']

            # Assign color for scatter plot
            parity_index = list(set(region_parity_map[region])).index(parity)
            plt.scatter(dim_range, milk_predictions,
                        label=f'Predicted Milk - Parity {parity} in {region}',
                        color=region_colors[region][parity_index], s=10)

            # Annotate each dot
            for x, y in zip(dim_range, milk_predictions):
                plt.annotate(f'{float(y):.2f}', (x, y), textcoords="offset points", xytext=(5, 5), ha='center')

        plt.title('Predicted Milk Yield vs Days in Milk')
        plt.xlabel('Days in Milk')
        plt.ylabel('Milk Yield (Kg)')
        plt.legend(fontsize= 8, loc='upper right')
        plt.grid(False)
        plt.show()


# Final MilkBot_Visuals Function, which integrate all above functions
def milkbot_prediction_visuals(region=['eu','us'], parity = ['1','2','3+'], dim_range=np.arange(1,306,1)):
    response = predict_milk_yield_for_region(region, parity, dim_range)
    return milk_yield_plot(response)
