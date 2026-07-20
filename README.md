# Spatial-Temporal-Multi-Tasking-Model
A transformer-based foundation model designed to learn spatial-temporal patterns from large-scale transportation data using self-supervised pretraining.

#Problem Statement
Existing foundation models have achieved success in NLP, but a relatively low amount of work has been done related to spatial-temporal data. This project aims to develop a reusable pretrained model to improve performance on downstream tasks like traffic forecasting and route analysis.

#Dataset
The model was trained using the 2013 New York City Taxi Trip Dataset, consisting of approximately 14 million data points per month over the span of a year. Each record includes pickup/drop-off timestamps, GPS coordinates, trip duration, trip distance, passenger count, vendor information, and other trip metadata.

#Results
The pretrained transformer model substantially outperformed baseline models. Trip duration, pickup zone prediction, and drop-off zone predictions all increased in accuracy greatly after implementing the pretraining. These improvements demonstrate the effectiveness of pretraining for spatial-temporal learning.

#Key Findings
• Transformer-based foundation models can effectively learn spatial-temporal representations through self-supervised pretraining.
• Pretraining dramatically improves downstream task performance compared to training from scratch.
• Mask-and-recovery training enables the model to capture meaningful spatial and temporal relationships.
• Foundation models show strong potential for applications such as traffic prediction, transportation optimization, ride-sharing systems, and urban planning.
