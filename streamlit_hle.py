import streamlit as st
import pandas as pd
import plotly.express as px


def render_category_bin(df):
    cat_counts = df['category'].value_counts().reset_index()
    cat_counts.columns = ['category', 'count']
    cat_counts['percent'] = cat_counts['count'] / cat_counts['count'].sum() * 100

    fig_cat = px.pie(
        cat_counts,
        names="category",
        values="count",
        color="category",
        labels={"category": "类别", "count": "数量"},
        height=500,
        hole=0.3,
        hover_data=["category", "count", "percent"]
    )
    fig_cat.update_traces(
        textinfo='percent+label',
        hovertemplate="<b>%{label}</b><br>数量: %{value}<br>占比: %{percent:.2%}<extra></extra>"
    )
    fig_cat.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        plot_bgcolor="#fff",
        paper_bgcolor="#fff",
        font=dict(size=14),
    )
    st.plotly_chart(fig_cat, use_container_width=True, config={"displayModeBar": False})

def render_category_details(df):
    category_options = df['category'].unique().tolist()
    default_category = "Math" if "Math" in category_options else category_options[0]
    col1, col2 = st.columns([1,1])
    with col1:
        selected_category = st.selectbox("选择类别 (category)", options=category_options, index=category_options.index(default_category))

    df_cat = df[df['category'] == selected_category]

    # 处理 level1，将只出现一次的标签归为“长尾标签”
    level1_counts_raw = df_cat['level1'].value_counts().reset_index()
    level1_counts_raw.columns = ['level1', 'count']

    # 找到出现次数小于 count 的 level1 作为长尾问题
    long_tail_level1 = level1_counts_raw[level1_counts_raw['count'] <= 3]['level1'].tolist()

    # 新增一列，归类长尾标签
    df_cat_plot = df_cat.copy()
    df_cat_plot['level1_grouped'] = df_cat_plot['level1'].apply(lambda x: '长尾标签' if x in long_tail_level1 else x)

    # 重新统计归类后的 level1
    level1_counts = df_cat_plot['level1_grouped'].value_counts().reset_index()
    level1_counts.columns = ['level1', 'count']
    level1_counts['percent'] = level1_counts['count'] / level1_counts['count'].sum() * 100

    # 标记占比小于1%的 level1
    level1_counts["level1_display"] = level1_counts.apply(
        lambda row: f"{row['level1']} 🔍" if row["percent"] < 1 else row["level1"], axis=1
    )

    # level2 selectbox
    with col2:
        level1_options = level1_counts['level1'].tolist()
        default_level1 = level1_options[0]
        selected_level1 = st.selectbox("选择一级分类 (level1)", options=level1_options, index=0)

    fig_level1 = px.pie(
        level1_counts,
        names="level1_display",
        values="count",
        color="level1_display",
        labels={"level1_display": "一级分类", "count": "数量"},
        height=400,
        hole=0.3,
        hover_data=["level1", "count", "percent"]
    )
    fig_level1.update_traces(
        textinfo='percent',
        textposition='auto',  # 自动决定标签位置，能写在内部就写在内部
        pull=[0.08 if row["percent"] < 1 else 0 for _, row in level1_counts.iterrows()],  # 占比<1%的拉出
        hovertemplate="<b>%{label}</b><br>数量: %{value}<br>占比: %{percent:.2%}<extra></extra>",
        # 限制引导线长度，避免超出高度
        insidetextorientation='auto',
        marker=dict(line=dict(color='#fff', width=1)),
        automargin=True
    )
    fig_level1.update_layout(
        uniformtext_minsize=10,
        uniformtext_mode='hide',
        margin=dict(l=20, r=20, t=30, b=20),
        plot_bgcolor="#fff",
        paper_bgcolor="#fff",
        font=dict(size=14),
        showlegend=True,  # 保留右侧图例
        height=400,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02
        )
    )
    # 饼图上只显示大于1%的标签
    fig_level1.update_traces(
        texttemplate=[
            f"{row['level1']}: {row['percent']:.1f}%" if row["percent"] >= 3 else ""
            for _, row in level1_counts.iterrows()
        ]
    )
    fig_level1.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        plot_bgcolor="#fff",
        paper_bgcolor="#fff",
        font=dict(size=14),
    )
    st.plotly_chart(fig_level1, use_container_width=True, config={"displayModeBar": False})

    if (level1_counts["percent"] < 1).any():
        st.caption(
            """
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 1.5em;">🔍</span>
                <span style="font-size: 1.1em;">部分一级分类占比低于 1%，仅在悬浮时显示。</span>
            </div>
            """,
            unsafe_allow_html=True
        )


    # 注意：如果选择了“长尾标签”，则筛选所有原本属于长尾的 level1
    if selected_level1 == "长尾标签":
        df_level1 = df_cat[df_cat['level1'].isin(long_tail_level1)]
    else:
        df_level1 = df_cat[df_cat['level1'] == selected_level1]

    level2_options = df_level1['level2'].unique().tolist()
    level2_options_sorted = sorted(level2_options)
    # selected_level2 = st.multiselect(
    #     "选择二级分类 (level2)",
    #     options=level2_options_sorted,
    #     default=level2_options_sorted
    # )

    # df_filtered = df_level1[df_level1['level2'].isin(selected_level2)]
    df_filtered = df_level1

    return df_filtered

def render_category_excels(df_filtered):
    # 3. 展示数据表格
    st.subheader("详细数据")
    st.dataframe(
        df_filtered[["category", "question", "level1", "level2"]],
        use_container_width=True
    )

# 新增：多模态分类柱状图和饼状图分析
def render_multimodal_bar(df2):
    # 统计每个category下is_multimodal的True/False数量
    bar_data = df2.groupby(['category', 'is_multimodal']).size().reset_index(name='count')
    # 保证is_multimodal为bool类型
    bar_data['is_multimodal'] = bar_data['is_multimodal'].astype(str)
    fig = px.bar(
        bar_data,
        x="category",
        y="count",
        color="is_multimodal",
        barmode="group",
        labels={"category": "类别", "count": "数量", "is_multimodal": "是否多模态"},
        height=500,
        text="count"
    )
    # 将数据标签显示在柱状图上方
    fig.update_traces(textposition='outside')
    fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        plot_bgcolor="#fff",
        paper_bgcolor="#fff",
        font=dict(size=14),
        legend_title_text="是否多模态"
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

def render_multimodal_pie_and_table(df2):
    # 只保留多模态数据
    df_mm = df2[df2['is_multimodal'] == True].copy()
    if df_mm.empty:
        st.info("没有 is_multimodal=True 的数据。")
        return

    # category选择
    category_options = df_mm['category'].unique().tolist()
    default_category = category_options[0]
    selected_category = st.selectbox("选择类别 (category)", options=category_options, index=0, key="mm_category")
    df_cat = df_mm[df_mm['category'] == selected_category]

    # 处理 level1，将只出现一次的标签归为“长尾标签”
    level1_counts_raw = df_cat['level1'].value_counts().reset_index()
    level1_counts_raw.columns = ['level1', 'count']
    long_tail_level1 = level1_counts_raw[level1_counts_raw['count'] <= 3]['level1'].tolist()
    df_cat_plot = df_cat.copy()
    df_cat_plot['level1_grouped'] = df_cat_plot['level1'].apply(lambda x: '长尾标签' if x in long_tail_level1 else x)
    level1_counts = df_cat_plot['level1_grouped'].value_counts().reset_index()
    level1_counts.columns = ['level1', 'count']
    level1_counts['percent'] = level1_counts['count'] / level1_counts['count'].sum() * 100
    level1_counts["level1_display"] = level1_counts.apply(
        lambda row: f"{row['level1']} 🔍" if row["percent"] < 1 else row["level1"], axis=1
    )

    fig_level1 = px.pie(
        level1_counts,
        names="level1_display",
        values="count",
        color="level1_display",
        labels={"level1_display": "一级分类", "count": "数量"},
        height=400,
        hole=0.3,
        hover_data=["level1", "count", "percent"]
    )
    fig_level1.update_traces(
        textinfo='percent',
        textposition='auto',
        pull=[0.08 if row["percent"] < 1 else 0 for _, row in level1_counts.iterrows()],
        hovertemplate="<b>%{label}</b><br>数量: %{value}<br>占比: %{percent:.2%}<extra></extra>",
        insidetextorientation='auto',
        marker=dict(line=dict(color='#fff', width=1)),
        automargin=True
    )
    fig_level1.update_layout(
        uniformtext_minsize=10,
        uniformtext_mode='hide',
        margin=dict(l=20, r=20, t=30, b=20),
        plot_bgcolor="#fff",
        paper_bgcolor="#fff",
        font=dict(size=14),
        showlegend=True,
        height=400,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02
        )
    )
    fig_level1.update_traces(
        texttemplate=[
            f"{row['level1']}: {row['percent']:.1f}%" if row["percent"] >= 3 else ""
            for _, row in level1_counts.iterrows()
        ]
    )
    fig_level1.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        plot_bgcolor="#fff",
        paper_bgcolor="#fff",
        font=dict(size=14),
    )
    st.plotly_chart(fig_level1, use_container_width=True, config={"displayModeBar": False})

    if (level1_counts["percent"] < 1).any():
        st.caption(
            """
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 1.5em;">🔍</span>
                <span style="font-size: 1.1em;">部分一级分类占比低于 1%，仅在悬浮时显示。</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    # level2 selectbox
    level1_options = level1_counts['level1'].tolist()
    default_level1 = level1_options[0]
    selected_level1 = st.selectbox("选择一级分类 (level1)", options=level1_options, index=0, key="mm_level1")
    if selected_level1 == "长尾标签":
        df_level1 = df_cat[df_cat['level1'].isin(long_tail_level1)]
    else:
        df_level1 = df_cat[df_cat['level1'] == selected_level1]

    level2_options = df_level1['level2'].unique().tolist()
    level2_options_sorted = sorted(level2_options)
    # selected_level2 = st.multiselect(
    #     "选择二级分类 (level2)",
    #     options=level2_options_sorted,
    #     default=level2_options_sorted,
    #     key="mm_level2"
    # )

    # df_filtered = df_level1[df_level1['level2'].isin(selected_level2)]
    df_filtered = df_level1

    # 展示数据表格
    st.subheader("详细多模态数据")
    st.dataframe(
        df_filtered[["category", "question", "level1", "level2"]],
        use_container_width=True
    )

def load_dfs():
    # 读取数据
    df = pd.read_json("datasets_v1/HLE_主题分类_合并结果.jsonl", lines=True)
    df2 = pd.read_json("datasets_v1/HLE_模态分类_合并结果.jsonl", lines=True)
    return df, df2
    
def page2() -> None:
    
    df, df2 = load_dfs()
    st.title("HLE 主题分类数据分析")

    with st.sidebar:
        analysis_mode = st.radio(
            "👁️ 选择分析模式",
            options=["原始分类分布", "主题分类分析", "多模态分类分析"],
            index=1,
            key="analysis_mode_radio",
            horizontal=False
        )
    if analysis_mode == "原始分类分布":
        st.subheader("原始分类分布")
        render_category_bin(df)
    elif analysis_mode == "主题分类分析":
        st.subheader("主题分类分析")
        df_filtered = render_category_details(df)
        render_category_excels(df_filtered)
    else:
        st.subheader("多模态分布分析")
        render_multimodal_bar(df2)
        st.divider()
        st.subheader("多模态细分分析")
        render_multimodal_pie_and_table(df2)


    # with st.expander("原始分类分布（饼状图）", expanded=True):
    #     render_category_bin(df)
    # with st.expander("原始分类细分分析（饼状图+表格）", expanded=True):
    #     df_filtered = render_category_details(df)
    #     render_category_excels(df_filtered)

    # # 新增：多模态分类分析
    # # 读取多模态数据
    # df2 = pd.read_json("HLE_模态分类_合并结果.jsonl", lines=True)

    # with st.expander("多模态分布（柱状图）", expanded=True):
    #     render_multimodal_bar(df2)
    # with st.expander("多模态细分分析（饼状图+表格）", expanded=True):
    #     render_multimodal_pie_and_table(df2)

if __name__ == "__main__":
    page2()
