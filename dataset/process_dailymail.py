#
#  Copyright (c) 2015, Facebook, Inc.
#  All rights reserved.
#
#  This source code is licensed under the BSD-style license found in the
#  LICENSE file in the root directory of this source tree. An additional grant
#  of patent rights can be found in the PATENTS file in the same directory.
#
#  Author: Alexander M Rush <srush@seas.harvard.edu>
#          Sumit Chopra <spchopra@fb.com>
#          Jason Weston <jase@fb.com>

#/usr/bin/env python

import sys
import os
import re
import gzip
#@lint-avoid-python-3-compatibility-imports

# Strip off .gz ending
end = "/".join(sys.argv[1].split("/")[-2:])[:-len(".summary")] + ".txt"

out = open(sys.argv[2] + end, "w")

f = open(sys.argv[1], "r")

# FIX: Some parses are mis-parenthesized.
def fix_paren(parse):
    if len(parse) < 2:
        return parse
    if parse[0] == "(" and parse[1] == " ":
        return parse[2:-1]
    return parse

def get_words(parse):
    words = []
    for w in parse.split():
        if w[-1] == ')':
            words.append(w.strip(")"))
            if words[-1] == ".":
                break
        else:
            words.append(w)
    return words

def remove_digits(parse):
    return re.sub(r'\d', '#', parse)


s = f.read().split("\n\n")
title_parse = remove_digits(fix_paren(re.sub("\n", " ", s[2]).strip()))
article_parse = "(TOP " + remove_digits(fix_paren(re.sub("\.\.", "", re.sub("\t\t\t\d[\n]*", ' ', s[1])).strip())) + ")"

# title_parse \t article_parse \t title \t article
print >>out, "\t".join([title_parse, article_parse,
                        " ".join(get_words(title_parse)),
                        " ".join(get_words(article_parse)),
                        "\n"])
