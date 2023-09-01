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

df <- read_csv('/home/otto/data/atmos-flux-data/output/atmos-fluxnet-20230111.csv')

df %>% summarise(FCH4tot = sum(FCH4, na.rm = T))

df %>% group_by(yday(date))

daily <- df %>%
  mutate(day_col = date(df$date)) %>%
  group_by(day_col) %>%
  summarize(TA_EPdaymean = mean(TA_EP, na.rm = T),
            LEdaysum = sum(LE, na.rm = T),
            FCdaysum = sum(FC, na.rm = T) * 18000 / 10e3,
            FCH4daysum = sum(FCH4, na.rm = T) * 18000 / 10e3
            )
daily

p1 <- daily %>% 
  ggplot(aes(x=day_col, y = TA_EPdaymean)) +
  geom_point(stat = 'identity', size = 2.5) +
  labs(x = 'date', y = 'TA (C)') +
  theme_minimal() +
  theme(legend.position = 'none', 
        panel.grid.major.x = element_blank(),
        # panel.grid.minor.y = element_line(),
        axis.title.x = element_blank(),
        axis.text.x = element_blank(),
        # aspect.ratio = 0.5
        plot.background = element_rect(color = 'white')
  )
p1

p2 <- daily %>% 
  ggplot(aes(x=day_col, y = LEdaysum)) +
  geom_bar(stat = 'identity', fill = 'black') +
  labs(x = 'date', y = 'LE (W m-2)') +
  theme_minimal() +
  theme(legend.position = 'none', 
        panel.grid.major.x = element_blank(),
        # panel.grid.minor.y = element_line(),
        axis.title.x = element_blank(),
        axis.text.x = element_blank(),
        # aspect.ratio = 0.5
        plot.background = element_rect(color = 'white')
  )
p2

p3 <- daily %>% 
  ggplot(aes(x=day_col, y = FCdaysum)) +
  geom_bar(stat = 'identity', fill = 'black') +
  labs(x = 'date', y = 'FC (mmol m-2)') +
  theme_minimal() +
  theme(legend.position = 'none', 
        panel.grid.major.x = element_blank(),
        # panel.grid.minor.y = element_line(),
        axis.title.x = element_blank(),
        axis.text.x = element_blank(),
        # aspect.ratio = 0.5
        plot.background = element_rect(color = 'white')
  )
p3


p4 <- daily %>% 
  ggplot(aes(x=day_col, y = FCH4daysum)) +
  geom_bar(stat = 'identity', fill = 'black') +
  labs(x = 'date', y = 'FCH4 (mmol m-2)') +
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

p1 / p2 / p3 / p4

ggsave(paste0(fr, 'output/atmos_dailyfluxes_', Sys.Date(), '.png'),
       height = 5.65*1.8, width = 10*1.8, units = c("in"))

# 
# df %>%
#   mutate(date_col = date(df$date)) %>%
#   group_by(date_col) %>%
#   summarize(TA_EPday = sum(df$TA_EP),
#             LEday = sum(df$LE),
#             FCday = sum(df$FC),
#             FCH4day = sum(df$FCH4)
#             )
# 
# palYlGn9 <- brewer.pal(9, 'YlGn')
# palTemp <- colorRampPalette(c(palYlGn9[5], palYlGn9[9]))(8)
# palGnBu9 <- brewer.pal(9, 'Blues')
# palBor <- colorRampPalette(c(palGnBu9[4], palGnBu9[9]))(11)
# palBuPu9 <- brewer.pal(9, 'BuPu')
# palPol <- colorRampPalette(c(palBuPu9[6], palBuPu9[9]))(2)
# # palPol <- palBuPu9
# kgpalette <- c(rev(brewer.pal(7, 'YlOrRd')), 
#                palTemp, 
#                palBor, 
#                palPol)
# 
# 
# 
# kgasia_sumzonal <- read_csv('/home/otto/git/repch4/v2/zonalsum_kgclim_2022-09-15_totarea.csv') %>% 
#   mutate_at(vars(kgclim), factor)
# kgasia_sumzonal
# 
# upch4_sumzonal <- read_csv('/home/otto/git/repch4/v2/zonalsum_upch4_2022-09-15_rougharea.csv') %>% 
#   mutate_at(vars(kgclim), factor)
# 
# wad2m_sumzonal <- read_csv('/home/otto/git/repch4/v2/zonalsum_wad2m_2022-09-15.csv') %>%
#   mutate_at(vars(kgclim), factor)
# 
# 
# wad2m_totarea <- cellSize(wad2masia1km)
# wad2m_totwetarea<- wad2masia1km * wad2m_totarea
# all.equal(wad2m_totwetarea, kgasia)
# wad2m_sumzonal_wetarea <- zonal(wad2m_totwetarea, kgasia, fun='sum', na.rm=TRUE) %>% 
#   `colnames<-`(c('kgclim', months)) %>% 
#   mutate_at(vars(kgclim), factor) %>% 
#   pivot_longer('jan':'dec', names_to = 'month', values_to = 'areawet_m2') %>% 
#   mutate(month = factor(month, levels = months))
# wad2m_sumzonal_wetarea
# # write_csv(wad2m_sumzonal_wetarea, paste0('/home/otto/git/repch4/v2/zonalsum_wad2m_', Sys.Date(), '_wetarea.csv'))
# 
# wad2m_totarea
# wad2m_sumzonal_totarea <- terra::zonal(wad2m_totarea, kgasia, fun='sum', na.rm=T) %>% 
#   `colnames<-`(c('kgclim', months)) %>%
#   mutate_at(vars(kgclim), factor) %>% 
#   pivot_longer('jan':'dec', names_to = 'month', values_to = 'areatot_m2') %>% 
#   mutate(month = factor(month, levels = months))
# # write_csv(wad2m_sumzonal_totarea, paste0('/home/otto/git/repch4/v2/zonalsum_wad2m_', Sys.Date(), '_totarea.csv'))
# 
# wad2m_sumzonal <- inner_join(wad2m_sumzonal_totarea, wad2m_sumzonal_wetarea, by = c('kgclim', 'month'))
# wad2m_sumzonal
# 
# # write_csv(wad2m_sumzonal, paste0('/home/otto/git/repch4/v2/zonalsum_wad2m_', Sys.Date(), '.csv'))
# 
# annual <- wad2m_sumzonal %>%
#   inner_join(upch4_sumzonal, by=c('kgclim', 'month')) %>%
#   group_by(kgclim) %>% 
#   summarize(TgCH4year = sum(TgCH4month),
#             min_TgCH4 = min(TgCH4month),
#             max_TgCH4 = max(TgCH4month),
#             seasonality_CH4 = min_TgCH4 / max_TgCH4,
#             min_areawet_km2 = min(areawet_m2) / 10e6,
#             max_areawet_km2 = max(areawet_m2) / 10e6, 
#             seasonality_wet = max_areawet_km2 / min_areawet_km2) %>% 
#   inner_join(kgasia_sumzonal, by = 'kgclim') %>% 
#   mutate(kgpal = as.factor(kgpalette)) %>% 
#   arrange(desc(TgCH4year))
# 
# annual$kgclim <- factor(annual$kgclim, labels = kgzones$code[c(seq(1, 9, 1), seq(11, 18, 1), seq(20, 30, 1))])
# 
# # write_csv(annual, paste0('/home/otto/git/repch4/v2/outputs/asia_annual_', Sys.Date(), '_rougharea.csv'))
# 
