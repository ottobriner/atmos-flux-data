#####
library(gtools)
library(ggplot2)
library(dplyr)
library(plotly)
# library(DescTools)

#The output files to be compiled are  "ATMOS"

#CODE:
#############################################################################################
####
# 1. Reading all the files
###
# clear memory

rm(list=ls())

# set path
#path <- "./Flux-tower/EP_outputs"
# path <- "/Users/marionnyberg/Google\ Drive/Micromet\ Lab/Projects/2019-Burns\ Bog\ 2/Flux-tower/EP_outputs"
path <- "/home/otto/data/atmos-flux-data/processedEP/EP_outputs/"
# path <- "/home/otto/data/atmos-flux-data/output/EP_outputs/"


# List only full_output files
raw.files <- list.files(path = path, pattern = "full_output", recursive = TRUE)
raw.data <- data.frame()

for(i in 1:length(raw.files)) {
  # Get header names
  names_temp <- names(read.csv(paste(path,"/",raw.files[i],sep=""),skip=1,sep=",",header=TRUE,dec="."))
	
	# Load data & apply header names
	temp <- read.csv(paste(path,"/",raw.files[i],sep=""),skip=3,header=FALSE) #skip=3 means skip the first 3 rows of the file
	names(temp) <- names_temp
	
	# Append to file
	raw.data <- smartbind(raw.data,temp, fill = "NA")
}

####
# 2. Creating a Timestamp with date and time variable to order the data by date-time
###
raw.data$Timestamp<-as.POSIXct(paste(raw.data$date, raw.data$time), format="%Y-%m-%d %H:%M", tz = "Etc/GMT+6")

data.ordered<-raw.data[order(raw.data$Timestamp, decreasing = FALSE ),]

data.ordered[1,1:5]
data.ordered[nrow(data.ordered),1:5]

####
#3. GAPFILLING ROWS BY GENERATING A NEW FILE  (Adapted from my "Gapfilling incomplete Time series" script)
#####

ts<-data.ordered

#to estimate necessary parameters to generate the new empty dataframe with complete rows 
begining<-as.numeric(ts$Timestamp[1])                           #to find the number that is represented by the first data of our time series
as.POSIXct(begining,origin="1970-01-01 00:00:00",tz="Etc/GMT+6") #to confirm that the begining is the right one
ts$Timestamp[1]

ts[ts$filename == "not_enough_data", 1:10]


Ystart<-as.integer(as.character(ts[1,ncol(ts)], "%Y"))
Yend<-as.integer(as.character(ts[nrow(ts),ncol(ts)], "%Y"))
Dstart<-as.POSIXlt(ts[1,ncol(ts)])$yday+1
Dend<-as.POSIXlt(ts[nrow(ts),ncol(ts)])$yday+1

Ystart
Yend
Dstart
Dend

Ndays <- as.numeric((difftime(ts$Timestamp[nrow(ts)],ts$Timestamp[1], "Etc/GMT+6",
															units = c("days"))), units="days")
Ndays

#To generate a serie of all minutes in a day:
Tsteps<-begining+seq(from=0,to=((Ndays)*(60*60*24)),by=(30*60)) # half an hours data from the begining date to Ndays +1 (that way I assure all measured data are included in the new file)
DATE<-as.POSIXct(Tsteps,origin="1970-01-01 00:00:00",tz="Etc/GMT+6")

# Make sure time series is continuoue
plot(diff(DATE))

#Confirming length of the new Time string 
#(THE NEW FILE SHOULD END SAME TIME THE TIME SERIES START AND ONE DAY AFTER THE MEASUREMENTS TIME SERIE)
DATE[1]
DATE[length(DATE)]
ts$Timestamp[1]
ts$Timestamp[length(ts$Timestamp)]

#GENERATING A NEW DATA FRAME WITH CONTINUOUS TIME STEPS and data from THE ORIGINAL ONE
cont.DS<-as.data.frame(DATE)
cont.DS[,c("DATE",names(ts)[1:(length(names(ts)))-1])]<-NA
cont.DS$DATE<-DATE

#FILLING THE NEW DATAFRAME WITH DATA FROM THE ORIGINAL DATA FRAME 
for(i in 2:ncol(cont.DS)){  
	cont.DS[,i]<-ts[pmatch(cont.DS$DATE,ts$Timestamp),i-1]  
	#pmatch look for the observation rows when time columns of both (old and new) dataframes match
} 

####
# 4. Adding local time
###
library(lubridate)
library(zoo)

date_loca <- ymd_hms(cont.DS$DATE, tz="America/Chicago")
date_local<-as.POSIXlt(date_loca,tz="America/Chicago")

for (i in 1:nrow(cont.DS)){
	cont.DS$Year_local[i]<-as.integer(as.character(date_local[i],"%Y"))
	cont.DS$jday_local[i]<-as.POSIXlt(date_local[i])$yday+1
	cont.DS$month_local[i]<-as.numeric(format(date_local[i],"%m"))
	cont.DS$hour_local[i]<-as.integer(as.character(date_local[i],"%H"))
	cont.DS$min_local[i]<-sprintf("%02s",as.integer(as.character(date_local[i],"%M")))  #sprintf function converts 0 in 00 to be pasted with hour to generate local time
	cont.DS$time_local[i]<-paste(cont.DS$hour_local[i],cont.DS$min_local[i],sep=":")
	day_portion<-ifelse(cont.DS$min_local[i]=="00",as.numeric(cont.DS$hour_local[i]),as.numeric(cont.DS$hour_local[i])+0.5)
	cont.DS$DOY_local[i]<-cont.DS$jday_local[i]+(day_portion*2*0.02)
}


####
# 5. Replacing -9999 by NA
###
cont.DS[cont.DS == -9999] <- NA

## Plot key variables
plot_ly(data = cont.DS, x = ~DATE, y = ~LE, name = 'LE', type = 'scatter', mode = 'lines') %>%
	add_trace(y = ~H, name = 'H', mode = 'lines') %>% 
  toWebGL()

plot_ly(data = cont.DS, x = ~DATE, y = ~ch4_flux, name = 'CH4', type = 'scatter', mode = 'lines') %>% 
  toWebGL()
plot_ly(data = cont.DS, x = ~DATE, y = ~co2_flux, name = 'CO2', type = 'scatter', mode = 'lines') %>% 
  toWebGL()
# plot_ly(data = cont.DS, x = ~DATE, y = ~SWIN, name = 'SWIN', type = 'scatter', mode = 'markers') %>%
#   toWebGL()

####
# 5. Saving the data
###

cont.DS$date <- as.Date(cont.DS$DATE) #already got this one from the answers above

cont.DS$time <- format(as.POSIXct(cont.DS$DATE) ,format = "%H:%M") 


write.csv(cont.DS,paste('/home/otto/data/atmos-flux-data/processed/ATMOS_L1_', Sys.Date(), '.csv',sep =''),row.names=FALSE)


