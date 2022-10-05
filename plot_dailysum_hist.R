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

df <- read_csv(paste0(fr, 'output/atmos-fluxnet-20221001.csv'))

df %>% summarise(FCH4tot = sum(FCH4, na.rm = T))

df %>% group_by(yday(date))

daily <- df %>%
  mutate(day_col = date(df$date)) %>%
  group_by(day_col) %>%
  summarize(TA_EPdaymean = mean(TA_EP, na.rm = T),
            Hdaysum = sum(H, na.rm = T),
            LEdaysum = sum(LE, na.rm = T),
            FCdaysum = sum(FC, na.rm = T) * 18000 / 10e3,
            FCH4daysum = sum(FCH4, na.rm = T) * 18000 / 10e3
            )
daily

means <- daily %>% 
  summarize(Hdaysum_mean = mean(Hdaysum, na.rm = T),
            LEdaysum_mean = mean(LEdaysum, na.rm = T),
            FCdaysum_mean = mean(FCdaysum, na.rm = T),
            FCH4daysum_mean = mean(FCH4daysum, na.rm = T))
means

# bw = 50
ymax1 = 12000
ymin1 = -3000
ymax2 = 1500
ymin2 = -1500
xmax1 = 25
xmax2 = 42

p1 <- daily %>% 
  ggplot(aes(y = Hdaysum)) +
  geom_histogram(fill = 'black') +
  # geom_tile() +
  lims(x = c(0, xmax1), y = c(ymin1, ymax1)) +
  labs(y = 'H (W m-2)') +
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

p2 <- daily %>% 
  ggplot(aes(y = LEdaysum)) +
  geom_histogram(fill = 'black') +
  labs(y = 'LE (W m-2)') +
  lims(x = c(0, xmax1), y = c(ymin1, ymax1)) +
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

p3 <- daily %>% 
  ggplot(aes(y = FCdaysum)) +
  geom_histogram(fill = 'black') +
  lims(x = c(0, xmax2), y = c(ymin2, ymax2)) +
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


p4 <- daily %>% 
  ggplot(aes(y = FCH4daysum)) +
  geom_histogram(fill = 'black') +
  lims(x = c(0, xmax2), y = c(ymin2, ymax2)) +
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

# ggsave(paste0(fr, 'output/atmos_dailyfluxes_hist_withaug_lims_', Sys.Date(), '.png'),
#        height = 9, width = 12, units = c("in"), dpi = 600)
