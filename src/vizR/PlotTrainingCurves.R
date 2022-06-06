library(ggplot)
library(cowplot)
library(reshape2)

loss.data <- read.csv('./loss_curves.csv')

loss.plot.df <- melt(loss.data[, c('epoch', 'test_loss', 'train_loss')], id.vars=c('epoch'))

ggplot(loss.plot.df, aes(x=epoch, y=log10(value), color=variable)) +
    geom_point(alpha=0.5) +
    geom_line() + theme_cowplot() + background_grid()

ggsave('./fig/graph_attention/loss_curves.png', width=8, height=6)

cos.sim.df <- melt(loss.data[, c('epoch', 'test_cos_sim', 'train_cos_sim')], id.vars=c('epoch'))

ggplot(cos.sim.df, aes(x=epoch, y=value, color=variable)) +
  geom_point(alpha=0.5) + ylim(c(-0.6, 1)) +
  geom_line() + theme_cowplot() + background_grid()

ggsave('./fig/graph_attention/cosine_similarity_curves.png', width=8, height=6)