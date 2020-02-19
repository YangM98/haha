#!usr/bin/python
# -*- coding: utf-8 -*-
#author:ly
import random

from pyecharts.faker import  Faker
from pyecharts import options as opts
from pyecharts.charts import HeatMap


def heatmap_base() -> HeatMap:
    value = [[i, j, random.randint(0, 50)] for i in range(24) for j in range(7)]
    c = (
        HeatMap()
        .add_xaxis(Faker.clock)
        .add_yaxis("series0", Faker.week, value)
        .set_global_opts(
            title_opts=opts.TitleOpts(title="HeatMap-基本示例"),
            visualmap_opts=opts.VisualMapOpts(),
        )
    )
    return c
c = heatmap_base()
c.render("heat.html")