import matplotlib.pyplot as plt
def create_charts(grouped_data): 
    charts = {} 
    try:
        grouped = grouped_data.groupby('cluster') 
        for cluster, group in grouped: 
            plt.figure(figsize=(10, 6)) 
            for product in group['cluster'].unique(): 
                product_group = group[group['cluster'] == product] 
                plt.plot(product_group['year'], product_group['price'], label=product) 
            chart_filename = f"cluster_{cluster}_chart.png" 
            plt.title(f"Cluster {cluster} - Price vs. Year") 
            plt.xlabel("Year") 
            plt.ylabel("Price") 
            plt.legend() 
            plt.savefig(chart_filename) 
            plt.close() 
            charts[cluster] = chart_filename
    except Exception as e:
        print(f"Error creating charts: {e}")
    finally:
        return charts
