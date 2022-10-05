## Load Packages
library(tidyverse)
library(lubridate)
library(ggthemes)
library(RColorBrewer)
library(viridis)
library(scales)
library(patchwork)
# library(raster)
library(rasterVis)
library(sf)
library(rworldmap)
library(rworldxtra)
library(cleangeo)
library(terra)
library(tidyterra)

fr <- '/home/otto/data/atmos-flux-data/'

df <- read_csv(paste0(fr, 'output/atmos-fluxnet-20221001.csv'))%>%
  mutate(day_col = date(date))
df

# ymax1 = 12000
# ymin1 = -3000
# ymax2 = 1500
# ymin2 = -1500
# xmax1 = 25
# xmax2 = 42
bw = 5

p1 <- df %>% 
  ggplot(aes(y = H)) +
  geom_histogram(fill = 'black', binwidth = bw) +
  # geom_tile() +
  # lims(x = c(0, xmax1), y = c(ymin1, ymax1)) +
  labs(y = 'H (W m-2 s-1)') +
  # coord_flip() +
  theme_minimal() +
  theme(legend.position = 'none', 
        panel.grid.major.x = element_blank(),
        # panel.grid.minor.y = element_line(),
        # axis.title.x = element_blank(),
        # axis.text.x = element_blank(),
        # aspect.ratio = 0.5
        plot.background = element_rect(color = 'white')
  )
p1

p2 <- df %>% 
  ggplot(aes(y = LE)) +
  geom_histogram(fill = 'black', binwidth = bw) +
  labs(y = 'LE (W m-2)') +
  # lims(x = c(0, xmax1), y = c(ymin1, ymax1)) +
  # coord_flip() +
  theme_minimal() +
  theme(legend.position = 'none', 
        panel.grid.major.x = element_blank(),
        # panel.grid.minor.y = element_line(),
        # axis.title.x = element_blank(),
        # axis.text.x = element_blank(),
        # aspect.ratio = 0.5
        plot.background = element_rect(color = 'white')
  )
p2

p3 <- df %>% 
  ggplot(aes(y = FC)) +
  geom_histogram(fill = 'black', binwidth = bw) +
  # lims(x = c(0, xmax2), y = c(ymin2, ymax2)) +
  labs(y = 'FC (mmol m-2)') +
  # coord_flip() +
  theme_minimal() +
  theme(legend.position = 'none', 
        panel.grid.major.x = element_blank(),
        # panel.grid.minor.y = element_line(),
        # axis.title.x = element_blank(),
        # axis.text.x = element_blank(),
        # aspect.ratio = 0.5
        plot.background = element_rect(color = 'white')
  )
p3


p4 <- df %>% 
  ggplot(aes(y = FCH4)) +
  geom_histogram(fill = 'black', binwidth = bw) +
  # lims(x = c(0, xmax2), y = c(ymin2, ymax2)) +
  labs(y = 'FCH4 (mmol m-2)') +
  # coord_flip() +
  theme_minimal() +
  theme(legend.position = 'none', 
        panel.grid.major.x = element_blank(),
        # panel.grid.minor.y = element_line(),
        # axis.title.x = element_blank(),
        # axis.text.x = element_blank(),
        # aspect.ratio = 0.5
        plot.background = element_rect(color = 'white')
  )
p4

(p1 + p2) / (p3 + p4)

# ggsave(paste0(fr, 'output/atmos_fluxhist_', Sys.Date(), '.png'),
#        height = 9, width = 12, units = c("in"), dpi = 600)
