#
# Build utility to package scripts commonly run together, or
# scripts required to configure the environment appropriately.
#
# @author Rahul Dhodapkar <rahul.dhodapkar@yale.edu>
#

.PHONY: configure

configure:
	./src/configure/download_visium_sample_data.sh

