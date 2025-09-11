import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_deepresearch import page1
from streamlit_hle import page2


@st.dialog("查看权限验证", width="medium", dismissible=False)
def password_dialog():
    st.write("请输入访问密码：")
    password = st.text_input("密码", type="password", key="password_input")
    if st.button("验证"):
        # 使用 Streamlit secrets 进行密码校验
        correct_password = st.secrets.get("dashboard_password")
        if password == correct_password:
            st.session_state["seen_password"] = password
            st.session_state["password_verified"] = True
            st.success("验证成功，欢迎访问！")
            st.rerun()
        else:
            st.session_state["seen_password"] = ""
            st.session_state["password_verified"] = False
            st.error("密码错误，请重试。")


def initialze_global_sesstion_state():
    # 用一个 flag 控制 default_index 只设置一次
    if "menu_initialized" not in st.session_state:
        st.session_state["menu_initialized"] = False
    
    if "seen_password" not in st.session_state:
        st.session_state["seen_password"] = ""

def main() -> None:
    st.set_page_config(page_title="Datasets Dashboard", layout="wide", page_icon="🌳")

    initialze_global_sesstion_state()

    # 页面加载时自动弹出密码验证
    if "password_verified" not in st.session_state or not st.session_state["password_verified"]:
        password_dialog()

    page_list = ["Deep Research 统计", "HLE 数据分类与统计"]

    with st.sidebar:
        if not st.session_state["menu_initialized"]:
            # 首次渲染时传入 default_index
            selected_page = option_menu(
                "数据集整理",
                page_list,
                menu_icon="bi bi-columns-gap",
                icons=["bi bi-book", "bi bi-person-workspace"],
                default_index=0,  # 默认选中第一个
                styles={
                    "nav-link": {"--hover-color": "#E9EFFF"},
                    "nav-link-selected": {"background-color": "#6372FF"},
                },
                key="menu_selection",
            )
            st.session_state["menu_initialized"] = True
        else:
            # 之后的 rerun 不再传 default_index，让组件自己保持状态
            selected_page = option_menu(
                "数据集整理",
                page_list,
                menu_icon="bi bi-columns-gap",
                icons=["bi bi-book", "bi bi-person-workspace"],
                styles={
                    "nav-link": {"--hover-color": "#E9EFFF"},
                    "nav-link-selected": {"background-color": "#6372FF"},
                },
                key="menu_selection",
            )

    # 路由
    if selected_page == "Deep Research 统计":
        page1()
    elif selected_page == "HLE 数据分类与统计":
        page2()

        

    

    

    


if __name__ == "__main__":
    main()
