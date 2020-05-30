# YLR
This is my work trying to understand Graph Variational Auto-Encoders. This started as my final project for the [Deep Learning course at Gatech](https://github.com/sreycodes/CS4803) and I tried many experiments to get the best possible scores on the youtube dataset. The above code was written using <https://github.com/lucashu1/link-prediction> and <https://github.com/deezer/gravity_graph_autoencoders/tree/master/gravity_gae>.

## Dataset
The dataset gives us a graph structure of the youtube videos where an edge between two videos means they are related i.e they appear in the related list of that video. We get the top 20 such related videos and other characteristics like view count, rating, etc.

## Results and experiments
We couldn't run our experiments on the whole graph due to its large size therefore I decided to choose the top 1000 videos based on some metric. I tried using rating (top-1000-rates.out), degree (top-1000-deg.out) and views (top-1000-views.out) to get the most important videos in the graph and obtained the best score with the top 1000 viewed videos. We run our further experiments on this subset of the data.

- top-1000-views-orig.out - Using a normal VGAE  

The following experiments use an approach inspired by [SEAL](https://arxiv.org/pdf/1802.09691.pdf). My main takeaway from this paper was that link prediction and graph embeddings are heavily dependent on their immediate neighbours or second-degree neighbours at max. Therefore we choose random edges and add their 1-hop sub-graphs incrementally while training the model. This gives us a substantial increase in the ROC score and AP score.  

- top-1000-views-negative.out - Added false training edges
- top-1000-views-D-best.out - Multiplied adjacency matrix by degree matrix as explained in <https://arxiv.org/pdf/1609.02907.pdf>
- top-1000-views-wd.out - Make the adjacency matrix weighted and directed and used a Gravity inspired Decoder based on <https://arxiv.org/pdf/1905.09570.pdf>

