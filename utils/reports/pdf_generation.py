def prepare_header(doc_obj, header_str, font_size):
    doc_obj.set_font('Arial', 'B', font_size)
    title_width = doc_obj.w - doc_obj.l_margin - doc_obj.r_margin
    doc_obj.cell(title_width, 10, header_str, 1, 0, 'C')
    doc_obj.ln(20)


def insert_single_img(doc_obj, xy_pos, img_path, img_title, img_dims):
    x_pos, y_pos = xy_pos
    single_img_width, single_img_height = img_dims

    doc_obj.set_xy(x_pos, y_pos)
    doc_obj.set_font('Arial', size=10)
    doc_obj.cell(single_img_width, 10, img_title, 1, 0, 'C')
    doc_obj.image(img_path, x=x_pos, y=y_pos + 15, w=single_img_width, h=single_img_height)


def insert_images_grid(doc_obj, img_grid_paths, img_grid_headers):
    assert len(img_grid_paths) <= 4
    assert len(img_grid_headers) == len(img_grid_paths)

    width_available = doc_obj.w - doc_obj.l_margin - doc_obj.r_margin
    height_available = doc_obj.h - doc_obj.b_margin - doc_obj.t_margin

    space_between_imgs = 10
    single_img_width = width_available / 2 - space_between_imgs / 2
    single_img_height = int(0.75 * single_img_width)

    x_cell_11 = doc_obj.l_margin
    x_cell_12 = x_cell_11 + single_img_width + space_between_imgs

    x_cells = [x_cell_11, x_cell_12, x_cell_11, x_cell_12]

    y_cell_11 = doc_obj.t_margin + 10 + 20
    y_cell_21 = y_cell_11 + (height_available / 2) + space_between_imgs

    y_cells = [y_cell_11, y_cell_11, y_cell_21, y_cell_21]

    for i, (img_grid_path, img_grid_header) in enumerate(zip(img_grid_paths, img_grid_headers)):
        if img_grid_path is not None:
            x, y = x_cells[i], y_cells[i]

            insert_single_img(doc_obj, (x, y), img_grid_path, img_grid_header, (single_img_width, single_img_height))
