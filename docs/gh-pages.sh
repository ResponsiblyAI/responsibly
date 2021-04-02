#!/bin/bash
# http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail
IFS=$'\n\t'

buildDirectory=_build/html

# # get a clean master branch assuming
# git checkout master
# git pull origin master
# git clean -df
# git checkout -- .
# git fetch --all
#
# # build html docs from sphinx files
# sphinx-build -b html . "$buildDirectory"

# create or use orphaned gh-pages branch
branch_name=gh-pages

if [ $(git branch --list "$branch_name") ]
then
	git stash
	git checkout $branch_name
	git pull origin $branch_name
	#git stash apply
	# git checkout --stash  . # force git stash to overwrite added files
else
	exit 1
fi

if [ -d "$buildDirectory" ]
then
  cd ..
	ls | grep -v docs | grep -v CNAME | xargs rm -rf

	mv docs/_build/html/* .
  rm -rf docs
	git add .
	git commit -m "new pages version $(date)"
	git push origin gh-pages
	# github.com recognizes gh-pages branch and create pages
	# url scheme https//:[github-handle].github.io/[repository]
else
	echo "directory $buildDirectory does not exists"
fi

git checkout master
