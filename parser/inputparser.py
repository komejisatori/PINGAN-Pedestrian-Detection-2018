def write_line(num, outfile, content):
    for i in range(num):
        outfile.write('\t')
    outfile.write('%s\n' % content)


def parser(train_name, val_name):
    tab_num = 0
    person_flag = False
    cyclist_flag = False
    person_train = open('person_train.txt', 'w')
    person_trainval = open('person_trainval.txt', 'w')
    person_val = open('person_val.txt', 'w')
    cyclist_train = open('cyclist_train.txt', 'w')
    cyclist_trainval = open('cyclist_trainval.txt', 'w')
    cyclist_val = open('cyclist_val.txt', 'w')

    with open(train_name, 'r') as file:
        count = 0
        for line in file.readlines():
            arg_list = line.split()
            if len(arg_list) == 1:
                continue
            with open('VOC2007/%s.xml' % arg_list[0].split('.')[0], 'w') as out:
                tmp_index = 1
                out.write('<annotation>\n')
                tab_num = tab_num + 1
                out.write('\t<folder>VOC2007</folder>\n')
                out.write('\t<filename>%s</filename>\n\t<size></size>\n\t<segmented>0</segmented>\n' % arg_list[0])
                while tmp_index < len(arg_list):
                    out.write('\t<object>\n')
                    tab_num = tab_num + 1
                    name = 'person' if arg_list[tmp_index] == '1' else 'cyclist'
                    if name == 'person':
                        person_flag = True
                    else:
                        cyclist_flag = True

                    write_line(tab_num, out, '<name>%s</name>' % name)
                    write_line(tab_num, out, '<pose>Unspecified</pose>')
                    write_line(tab_num, out, '<truncated>0</truncated>')
                    write_line(tab_num, out, '<difficult>0</difficult>')
                    write_line(tab_num, out, '<bndbox>')
                    tab_num = tab_num + 1
                    write_line(tab_num, out, '<xmin>%s</xmin>' % arg_list[tmp_index + 1])
                    write_line(tab_num, out, '<xmin>%s</xmin>' % arg_list[tmp_index + 2])
                    write_line(tab_num, out,
                               '<xmax>%s</xmax>' % (int(arg_list[tmp_index + 1]) + int(arg_list[tmp_index + 3])))
                    write_line(tab_num, out,
                               '<xmax>%s</xmax>' % (int(arg_list[tmp_index + 1]) + int(arg_list[tmp_index + 4])))
                    tab_num = tab_num - 1
                    write_line(tab_num, out, '</bndbox>')
                    tab_num = tab_num - 1
                    write_line(tab_num, out, '</object>')
                    tmp_index = tmp_index + 5
                tab_num = tab_num - 1
                write_line(tab_num, out, '</annotation>')
            if person_flag:
                person_train.write('%s 1\n' % arg_list[0].split('.')[0])
                person_trainval.write('%s 1\n' % arg_list[0].split('.')[0])
            else:
                person_train.write('%s -1\n' % arg_list[0].split('.')[0])
                person_trainval.write('%s -1\n' % arg_list[0].split('.')[0])
            if cyclist_flag:
                cyclist_train.write('%s 1\n' % arg_list[0].split('.')[0])
                cyclist_trainval.write('%s 1\n' % arg_list[0].split('.')[0])
            else:
                cyclist_train.write('%s -1\n' % arg_list[0].split('.')[0])
                cyclist_trainval.write('%s -1\n' % arg_list[0].split('.')[0])
            person_flag = False
            cyclist_flag = False
            count = count + 1
            print(count)

    with open(val_name, 'r') as file:
        count = 0
        for line in file.readlines():
            arg_list = line.split()
            if len(arg_list) == 1:
                continue
            with open('VOC2007/%s.xml' % arg_list[0].split('.')[0], 'w') as out:
                tmp_index = 1
                out.write('<annotation>\n')
                tab_num = tab_num + 1
                out.write('\t<folder>VOC2007</folder>\n')
                out.write('\t<filename>%s</filename>\n\t<size></size>\n\t<segmented>0</segmented>\n' % arg_list[0])
                while tmp_index < len(arg_list):
                    out.write('\t<object>\n')
                    tab_num = tab_num + 1
                    name = 'person' if arg_list[tmp_index] == '1' else 'cyclist'
                    if name == 'person':
                        person_flag = True
                    else:
                        cyclist_flag = True

                    write_line(tab_num, out, '<name>%s</name>' % name)
                    write_line(tab_num, out, '<pose>Unspecified</pose>')
                    write_line(tab_num, out, '<truncated>0</truncated>')
                    write_line(tab_num, out, '<difficult>0</difficult>')
                    write_line(tab_num, out, '<bndbox>')
                    tab_num = tab_num + 1
                    write_line(tab_num, out, '<xmin>%s</xmin>' % arg_list[tmp_index + 1])
                    write_line(tab_num, out, '<xmin>%s</xmin>' % arg_list[tmp_index + 2])
                    write_line(tab_num, out,
                               '<xmax>%s</xmax>' % (int(arg_list[tmp_index + 1]) + int(arg_list[tmp_index + 3])))
                    write_line(tab_num, out,
                               '<xmax>%s</xmax>' % (int(arg_list[tmp_index + 1]) + int(arg_list[tmp_index + 4])))
                    tab_num = tab_num - 1
                    write_line(tab_num, out, '</bndbox>')
                    tab_num = tab_num - 1
                    write_line(tab_num, out, '</object>')
                    tmp_index = tmp_index + 5
                tab_num = tab_num - 1
                write_line(tab_num, out, '</annotation>')
            if person_flag:
                person_val.write('%s 1\n' % arg_list[0].split('.')[0])
                person_trainval.write('%s 1\n' % arg_list[0].split('.')[0])
            else:
                person_val.write('%s -1\n' % arg_list[0].split('.')[0])
                person_trainval.write('%s -1\n' % arg_list[0].split('.')[0])
            if cyclist_flag:
                cyclist_val.write('%s 1\n' % arg_list[0].split('.')[0])
                cyclist_trainval.write('%s 1\n' % arg_list[0].split('.')[0])
            else:
                cyclist_val.write('%s -1\n' % arg_list[0].split('.')[0])
                cyclist_trainval.write('%s -1\n' % arg_list[0].split('.')[0])
            person_flag = False
            cyclist_flag = False
            count = count + 1
            print(count)


parser('train_annotations.txt', 'val_annotations.txt')
