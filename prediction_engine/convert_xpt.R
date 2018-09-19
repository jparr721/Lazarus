library(readr)
library(SASxport)

setwd("~/ML/Lazarus/Diabetes-Data/csvs")

xptfiles <- list.files(pattern = "*.XPT")
csvfiles <- NULL

for (i in 1:length(xptfiles)) {
  xptfile <- xptfiles[i]
  file_name <- substr(xptfile, 1, nchar(xptfile) - 3)
  ext <- "csv"

  csvfile <- paste0(file_name, ext)
  csvfiles[i] <- paste0(file_name, ext)

  df <- read.xport(xptfile)
  write.csv(df, file = csvfile, row.names = FALSE)
  cat(c(paste0("[", i, "]"), csvfiles[i], '\n'))
}
