from asyncio.log import logger
import pickle
import json
import multiprocessing
import os
import re
import shutil
import subprocess
import traceback
import numpy as np
from tqdm import tqdm
from xml.etree.ElementTree import parse
from data.drain_bigfile import drain3
from bs4 import BeautifulSoup

def distribute_data(data, num_processes, process_id):
    # 生成等差数列
    indices = np.arange(process_id, len(data), num_processes)
    # 分配数据
    return [data[i] for i in indices]

def getJava(file_list:list, path):
    for file in os.listdir(path):
        file_path = os.path.join(path,file)
        if file[-5:]=='.java':
            file_list.append(file_path)
        elif os.path.isdir(file_path):
            getJava(file_list,file_path)
    return

def getJavaTrace(path, raw_path):
    trace_list = []
    for file in os.listdir(path):
        file_path = os.path.join(path,file)
        if file[-5:]=='.java':
            trace_list.append(file_path[len(raw_path)+1:].replace('/','.').replace('\\','.')[:-5])
        elif os.path.isdir(file_path):
            trace_list.extend(getJavaTrace(file_path, raw_path))
    return trace_list
            
def GetFilePath(module:str,test_coverage_map,my_logger):
    my_logger.info('generate mutation injection position')
    module_path = os.path.join('dataset','raw',module)
    process_module_path = os.path.join('dataset','process',module)
    if not os.path.exists(process_module_path):
        os.makedirs(process_module_path)
    rawData_path = os.path.join('dataset','rawData', module)
    if not os.path.exists(rawData_path):
        os.makedirs(rawData_path)
    insert_file_path = os.path.join(rawData_path,'mutation_position.txt')
    if 'hive' in module:
        file_path = os.path.join(module_path,'src','java')
    else:
        file_path = os.path.join(module_path,'src','main','java')
    file_list = []
    getJava(file_list, file_path)
    src_class = set([src_line.split('#')[0] for src_line in test_coverage_map.keys()])
    with open(insert_file_path,'w') as wf:
        for file in file_list:
            file_name = file[len(file_path)+1:-5].replace('\\','.').replace('/','.')
            if file_name not in src_class:
                continue
            else:
                flag = False
                for src_line in test_coverage_map.keys():
                    if file_name in src_line and len(test_coverage_map[src_line])>0:
                        for test_method in test_coverage_map[src_line]:
                            if 'ESTest' not in test_method:
                                flag = True
                            break
                        if flag:
                            break
                if not flag:
                    continue
            with open(file,'r',encoding='utf-8') as rf:
                line_num = len(rf.readlines())
            i = 5
            while i<line_num:
                wf.write(f'{file_name}#{i}\t1\n')
                i+=10
    return

def readMutant(module:str,my_logger):
    mutants_path = os.path.join('dataset/rawData',module,'mutants.json')
    my_logger.info('read mutants from %s' % mutants_path)
    mutant_info = json.load(open(mutants_path, 'r'))
    mutant_list = []
    for mutant in mutant_info:
        file_path = mutant['filename'][len('dataset/raw/')+len(module)+1:]
        file_line = mutant['line']
        mutant_code = mutant['code']
        mutant_list.append((file_path,file_line,mutant_code))
    return mutant_list

def collectTestMethod(module:str,my_logger):
    rawData_path = os.path.join('dataset','rawData',module)
    if not os.path.exists(rawData_path):
        os.makedirs(rawData_path)
    method_file = os.path.join('dataset','rawData',module,'testMethod.txt')
    if os.path.exists(method_file):
        return
    my_logger.info('collect test method.')
    raw_project_path  = os.path.join('dataset','raw',module.split('/')[0])
    execute_path = os.path.join('dataset','execute')
    if not os.path.exists(execute_path):
        os.makedirs(execute_path)
    execute_project_path = os.path.join(execute_path,module.split('/')[0]+'_method')
    if os.path.exists(execute_project_path):
        shutil.rmtree(execute_project_path)
    shutil.copytree(raw_project_path,execute_project_path)
    os.chdir(execute_project_path)
    if('/' in module):
        pl = '/'.join(module.split('/')[1:])
        surefire_path = os.path.join(pl,'target/surefire-reports')
        process = subprocess.run(["mvn", "clean", "test", "-pl", pl ,"-Drat.skip=true", "-Dcheckstyle.skip=true", "-Dlicense.skip=true", "-Dpmd.skip=true", "-Denforcer.skip=true"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        surefire_path = 'target/surefire-reports'
        process = subprocess.run(["mvn", "clean", "test", "-Drat.skip=true", "-Dcheckstyle.skip=true", "-Dlicense.skip=true", "-Dpmd.skip=true", "-Denforcer.skip=true"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode != 0:
        my_logger.error('{module} mvn test fail!')
        print(process.stdout,process.stderr)
        exit(-1)
    os.chdir('../../../')
    raw_testcase = []
    evo_testcase = []
    surefire_path = os.path.join(execute_project_path,surefire_path)
    pattern = re.compile(r'\.txt$')
    with open(method_file, 'w', encoding="utf-8") as write_file:
        for surefire_reports_file in os.listdir(surefire_path):
            if re.search(pattern,surefire_reports_file): # log file
                if surefire_reports_file[:-4] == 'null':
                    continue
                with open(os.path.join(surefire_path,surefire_reports_file),'r') as f:
                    for i in range(4):
                        f.readline()
                    for line in f.readlines():
                        if '(' in line:
                            method = line.strip().split('(')[0]
                        else:
                            method = line.strip().split()[0].split('.')[-1]
                        write_file.write(surefire_reports_file[:-4]+'#'+method+'\n')
                        if 'ESTest' in surefire_reports_file:
                            evo_testcase.append(method)
                        else:
                            raw_testcase.append(method)
    logger.info(f'raw testcase method number: {len(raw_testcase)}')
    logger.info(f'evosuite testcase method number: {len(evo_testcase)}')
    shutil.rmtree(execute_project_path,ignore_errors=True)

def testCoverMap(module,num_processes,my_logger):
    my_logger.info(f'execute test and collect coverage map between testcase and source code')
    process_path = os.path.join('dataset','process')
    coverMap = os.path.join('dataset/rawData',module,'cover_map.pkl')
    if os.path.exists(coverMap):
        with open(coverMap,'rb') as f:
            test_coverage_map = pickle.load(f)
        return test_coverage_map
    else:
        test_coverage_map = {}
        test_method_set = set()
        method_file = os.path.join('dataset','rawData',module,'testMethod.txt')
        with open(method_file,'r') as f:
            for line in f.readlines():
                if line.strip()!="":
                    test_method_set.add(line.strip())
        test_method_list = list(test_method_set)
        results = []
        pool2 = multiprocessing.Pool(num_processes)
        for process_id in range(num_processes):
            data_chunk = distribute_data(test_method_list, num_processes, process_id)
            result = pool2.apply_async(collect_cover_map,args=(module,data_chunk,process_id))
            results.append(result)
        pool2.close()
        pool2.join()
        for result in results:
            result = result.get()
            for mutant in result.keys():
                if mutant not in test_coverage_map:
                    test_coverage_map[mutant] = result[mutant]
                else:
                    for test_method in result[mutant]:
                        test_coverage_map[mutant].add(test_method)
        with open(coverMap,'wb') as f:
            pickle.dump(test_coverage_map,f)
        return test_coverage_map

def collect_cover_map(module,test_method_list,process_id):
    test_coverage_map = {}
    try:
        execute_path = os.path.join('dataset','execute')
        if not os.path.exists(execute_path):
            os.makedirs(execute_path)
        execute_coverage_project_path = os.path.join(execute_path,module.split('/')[0]+'_coverage_'+str(process_id))
        if os.path.exists(execute_coverage_project_path):
            shutil.rmtree(execute_coverage_project_path)
        raw_module_path = os.path.join('dataset/raw',module.split('/')[0])
        shutil.copytree(raw_module_path,execute_coverage_project_path)
        if('/' in module):
            pl = '/'.join(module.split('/')[1:])
            os.chdir(execute_coverage_project_path)
            # check if file is illegal
            for test_method in tqdm(test_method_list):
                try:
                    process = subprocess.run(["mvn", "clean", "test", "-Dtest="+test_method, "-pl", pl ,"-Drat.skip=true", "-Dcheckstyle.skip=true", "-Dlicense.skip=true", "-Dpmd.skip=true", "-Denforcer.skip=true"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    jacoco_path = os.path.join(pl,'target/site/jacoco-ut')
                    with open(os.path.join(jacoco_path,'index.html'), 'r') as f:
                        html = f.read()
                    soup = BeautifulSoup(html,'html.parser')
                    for package in soup.select('table tbody tr'):
                        link = package.select_one('td a')
                        if link is not None:  # 检查行中是否有链接
                            package_name = link.text.strip()
                            package_url = link['href']
                            # 检查文件是否包含覆盖信息
                            if 'index.html' in package_url:
                                with open(os.path.join(jacoco_path,package_url), 'r') as f:
                                    package_content = f.read()
                                package_soup = BeautifulSoup(package_content, 'html.parser')
                                for java_class in package_soup.select('table tbody tr'):
                                    link = java_class.select_one('td a')
                                    if link is not None:  # 检查行中是否有链接
                                        class_name = link.text.strip()
                                        class_name_url = link['href']
                                        # 检查文件是否包含覆盖信息
                                        if '.html' in class_name_url and '$' not in class_name_url:
                                            try:
                                                with open(os.path.join(jacoco_path,package_name,class_name_url[:-5]+'.java'+'.html'), 'r') as f:
                                                    class_content = f.read()
                                                class_soup = BeautifulSoup(class_content, 'html.parser')
                                                # 找到所有被覆盖的行
                                                for covered_line in class_soup.select('pre span.fc'):
                                                    line_number = covered_line.attrs['id']
                                                    cover_line = package_name+'.'+class_name+'#'+line_number
                                                    if cover_line not in test_coverage_map:
                                                        test_coverage_map[cover_line] = set()
                                                    test_coverage_map[cover_line].add(test_method)
                                            except Exception as e:
                                                traceback.print_exc()
                except Exception as e:
                    traceback.print_exc()
            os.chdir('../../../')
        else:
            os.chdir(execute_coverage_project_path)
            # check if file is illegal
            for test_method in tqdm(test_method_list):
                try:
                    process = subprocess.run(["mvn", "clean", "test", "-Dtest="+test_method ,"-Drat.skip=true", "-Dcheckstyle.skip=true", "-Dlicense.skip=true", "-Dpmd.skip=true", "-Denforcer.skip=true"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    jacoco_path = 'target/site/jacoco-ut'
                    with open(os.path.join(jacoco_path,'index.html'), 'r') as f:
                        html = f.read()
                    soup = BeautifulSoup(html,'html.parser')
                    for package in soup.select('table tbody tr'):
                        link = package.select_one('td a')
                        if link is not None:  # 检查行中是否有链接
                            package_name = link.text.strip()
                            package_url = link['href']
                            # 检查文件是否包含覆盖信息
                            if 'index.html' in package_url:
                                with open(os.path.join(jacoco_path,package_url), 'r') as f:
                                    package_content = f.read()
                                package_soup = BeautifulSoup(package_content, 'html.parser')
                                for java_class in package_soup.select('table tbody tr'):
                                    link = java_class.select_one('td a')
                                    if link is not None:  # 检查行中是否有链接
                                        class_name = link.text.strip()
                                        class_name_url = link['href']
                                        # 检查文件是否包含覆盖信息
                                        if '.html' in class_name_url and '$' not in class_name_url:
                                            try:
                                                with open(os.path.join(jacoco_path,package_name,class_name_url[:-5]+'.java'+'.html'), 'r') as f:
                                                    class_content = f.read()
                                                class_soup = BeautifulSoup(class_content, 'html.parser')
                                                # 找到所有被覆盖的行
                                                for covered_line in class_soup.select('pre span.fc'):
                                                    line_number = covered_line.attrs['id']
                                                    cover_line = package_name+'.'+class_name+'#'+line_number
                                                    if cover_line not in test_coverage_map:
                                                        test_coverage_map[cover_line] = set()
                                                    test_coverage_map[cover_line].add(test_method)
                                            except Exception as e:
                                                traceback.print_exc()
                except Exception as e:
                    traceback.print_exc()
            os.chdir('../../../')
        shutil.rmtree(execute_coverage_project_path,ignore_errors=True)
    except Exception as e:
        traceback.print_exc()
    return test_coverage_map

def filter_mutant(module, mutant_list, process_id, test_coverage_map, lock):
    compile_pass_mutant = []
    execute_path = os.path.join('dataset','execute')
    if not os.path.exists(execute_path):
        os.makedirs(execute_path)
    process_path = os.path.join('dataset','process')
    compile_pass_mutant_file = os.path.join('dataset/rawData',module,'compile_mutants.json')
    execute_clean_project_path = os.path.join(execute_path,module.split('/')[0]+'_clean')
    execute_project_path = os.path.join(execute_path,module.split('/')[0]+'_'+str(process_id))
    for mutant_info in tqdm(mutant_list):
        try:
            file_path = mutant_info[0]
            file_line = mutant_info[1]
            mutant_code = mutant_info[2]
            if 'src/main/java' in file_path:
                mutant_loc = file_path[14:-5].replace('/','.').replace('\\','.')+'#L'+str(file_line)
            else:
                mutant_loc = file_path[9:-5].replace('/','.').replace('\\','.')+'#L'+str(file_line)
            if mutant_loc not in test_coverage_map.keys() or len(test_coverage_map[mutant_loc])==0:
                continue
            if os.path.exists(execute_project_path):
                shutil.rmtree(execute_project_path)
            shutil.copytree(execute_clean_project_path,execute_project_path)
            if('/' in module):
                target_java_file_path = os.path.join(execute_path,module.split('/')[0]+'_'+str(process_id),'/'.join(module.split('/')[1:]),file_path)
            else:
                target_java_file_path = os.path.join(execute_path,module.split('/')[0]+'_'+str(process_id),file_path)
            os.remove(target_java_file_path)
            with open(target_java_file_path,'w') as f:
                f.write(mutant_code)
            os.chdir(execute_project_path)
            if('/' in module):
                pl = '/'.join(module.split('/')[1:])
                # check if file is illegal
                process = subprocess.run(["mvn", "compile", "-pl", pl, "-Drat.skip=true", "-Dcheckstyle.skip=true", "-Dlicense.skip=true", "-Dpmd.skip=true", "-Denforcer.skip=true"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if process.returncode == 0:
                    compile_pass_mutant.append(mutant_info)
            else:
                # check if file is illegal
                process = subprocess.run(["mvn", "compile", "-Drat.skip=true", "-Dcheckstyle.skip=true", "-Dlicense.skip=true", "-Dpmd.skip=true", "-Denforcer.skip=true"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if process.returncode == 0:
                    compile_pass_mutant.append(mutant_info)
            os.chdir('../../../')
            shutil.rmtree(execute_project_path,ignore_errors=True)
        except:
            continue
    with lock:
        with open(compile_pass_mutant_file, 'a', encoding='utf-8') as f:
            for mutant in compile_pass_mutant:
                json.dump(mutant,f)
                f.write('\n')
    return compile_pass_mutant

def log_collect(module, mutant_list, test_method_set, process_id, my_logger, lock, test_coverage_map):
    try:
        level_list = ['ALL','TRACE','DEBUG','INFO','WARN','ERROR','FATAL']
        execute_path = os.path.join('dataset','execute')
        if not os.path.exists(execute_path):
            os.makedirs(execute_path)
        process_path = os.path.join('dataset','process')
        execute_project_path = os.path.join(execute_path,module.split('/')[0]+'_'+str(process_id))
        execute_clean_project_path = os.path.join(execute_path,module.split('/')[0]+'_clean')
        for mutant_id, mutant_info in enumerate(mutant_list):
            my_logger.info(f'mutant id: {mutant_id} / {len(mutant_list)}')
            file_path = mutant_info[0]
            file_line = mutant_info[1]
            mutant_code = mutant_info[2]
            if os.path.exists(execute_project_path):
                shutil.rmtree(execute_project_path)
            shutil.copytree(execute_clean_project_path,execute_project_path)
            if('/' in module):
                target_java_file_path = os.path.join(execute_project_path,'/'.join(module.split('/')[1:]),file_path)
            else:
                target_java_file_path = os.path.join(execute_project_path,file_path)
            os.remove(target_java_file_path)
            with open(target_java_file_path,'w') as f:
                f.write(mutant_code)
            if 'src/main/java' in file_path:
                method_name = file_path[14:-5]
            else:
                method_name = file_path[9:-5]
            test_output_list = []
            if('/' in module):
                pl = '/'.join(module.split('/')[1:])
                surefire_path = os.path.join(pl,'target/surefire-reports')
                os.chdir(execute_project_path)
                # check if file is illegal
                with tqdm(total=len(list(test_method_set)),desc=f'mutant {mutant_id}/{len(mutant_list)}') as pbar:
                    for test_method in list(test_method_set):
                        if test_method not in test_coverage_map[method_name.replace('/','.').replace('\\','.')+'#L'+str(file_line)]:
                            pbar.update(1)
                            continue
                        try:
                            test_class = test_method.split('#')[0]
                            with open('test.log','w') as f:
                                process = subprocess.run(["mvn", "test", "-Dtest="+test_method, "-pl", pl ,"-Drat.skip=true", "-Dcheckstyle.skip=true", "-Dlicense.skip=true", "-Dpmd.skip=true", "-Denforcer.skip=true"], stdout=f, stderr=subprocess.PIPE, timeout=30 * 60)
                            trace_log = ""
                            with open('test.log','rb') as log_file:
                                for line in log_file:
                                    try:
                                        line = line.decode('utf-8')
                                        line_tokens = line.strip().split()
                                        if len(line_tokens)>0 and line_tokens[0]=='#####':
                                            # trace+=line_tokens[1]+'\n'
                                            trace_log += line_tokens[1] + '\n'
                                        elif len(line_tokens) > 2 and re.match(r'\d{4}\-\d{2}-\d{2}', line_tokens[0]):
                                            if line_tokens[2] not in level_list:
                                                continue
                                            # log_content += line
                                            trace_log += line
                                    except:
                                        continue
                            with open(os.path.join(surefire_path,test_class+'.txt'),'r') as test_txt:
                                file_content = test_txt.readlines()
                                result_line = file_content[3]
                                match = re.search(r"Tests run: (\d+), Failures: (\d+), Errors: (\d+), Skipped: (\d+)", result_line)
                                if match:
                                    tests_run = int(match.group(1))
                                    failures = int(match.group(2))
                                    errors = int(match.group(3))
                                    skipped = int(match.group(4))
                                if len(trace_log)>0:
                                    if failures + errors == 0:
                                        test_output_list.append({'trace_log':trace_log,'test':'normal','testcase':test_method})
                                    else:
                                        test_output_list.append({'trace_log':trace_log,'test':'abnormal','testcase':test_method})
                        except subprocess.TimeoutExpired:
                            my_logger.warning('mvn test timeout')
                        pbar.update(1)
                os.chdir('../../../')
                my_logger.info(f'mutant:{mutant_id} mvn test done.')
            else:
                surefire_path = 'target/surefire-reports'
                os.chdir(execute_project_path)
                # check if file is illegal
                with tqdm(total=len(list(test_method_set)),desc=f'mutant {mutant_id}/{len(mutant_list)}') as pbar:
                    for test_method in tqdm(list(test_method_set)):
                        if test_method not in test_coverage_map[method_name.replace('/','.').replace('\\','.')+'#L'+str(file_line)]:
                            continue
                        try:
                            test_class = test_method.split('#')[0]
                            with open('test.log','w') as f:
                                process = subprocess.run(["mvn", "test", "-Dtest="+test_method, "-Drat.skip=true", "-Dcheckstyle.skip=true", "-Dlicense.skip=true", "-Dpmd.skip=true", "-Denforcer.skip=true"], stdout=f, stderr=subprocess.PIPE, timeout=30 * 60)
                            trace_log = ""
                            with open('test.log','rb') as log_file:
                                for line in log_file:
                                    try:
                                        line = line.decode('utf-8')
                                        line_tokens = line.strip().split()
                                        if len(line_tokens)>0 and line_tokens[0]=='#####':
                                            # trace+=line_tokens[1]+'\n'
                                            trace_log += line_tokens[1] + '\n'
                                        elif len(line_tokens) > 2 and re.match(r'\d{4}\-\d{2}-\d{2}', line_tokens[0]):
                                            if line_tokens[2] not in level_list:
                                                continue
                                            # log_content += line
                                            trace_log += line
                                    except:
                                        continue
                            with open(os.path.join(surefire_path,test_class+'.txt'),'r') as test_txt:
                                file_content = test_txt.readlines()
                                result_line = file_content[3]
                                match = re.search(r"Tests run: (\d+), Failures: (\d+), Errors: (\d+), Skipped: (\d+)", result_line)
                                if match:
                                    tests_run = int(match.group(1))
                                    failures = int(match.group(2))
                                    errors = int(match.group(3))
                                    skipped = int(match.group(4))
                                if len(trace_log)>0:
                                    if failures + errors == 0:
                                        test_output_list.append({'trace_log':trace_log,'test':'normal','testcase':test_method})
                                    else:
                                        test_output_list.append({'trace_log':trace_log,'test':'abnormal','testcase':test_method})
                        except subprocess.TimeoutExpired:
                            my_logger.warning('mvn test timeout')
                        pbar.update(1)
                os.chdir('../../../')
                my_logger.info(f'mutant:{mutant_id} mvn test done.')
            log_file = os.path.join('dataset/rawData',module,'log.json')
            with lock:
                with open(log_file, 'a', encoding='utf-8') as f:
                    for test_output in test_output_list:
                        json.dump(test_output,f)
                        f.write('\n')
            shutil.rmtree(execute_project_path,ignore_errors=True)
    except Exception as e:
        traceback.print_exc()
    return
        
def executeProjectWithMutant(module:str,mutant_list:list,num_processes:int,test_coverage_map,my_logger):
    method_file = os.path.join('dataset','rawData',module,'testMethod.txt')
    level_list = ['ALL','TRACE','DEBUG','INFO','WARN','ERROR','FATAL']
    module_path = os.path.join('dataset','raw',module)
    test_method_set = set()
    with open(method_file,'r') as f:
        for line in f.readlines():
            if line.strip()!="":
                test_method_set.add(line.strip())
    execute_path = os.path.join('dataset','execute')
    if not os.path.exists(execute_path):
        os.makedirs(execute_path)
    execute_project_path = os.path.join(execute_path,module.split('/')[0])
    execute_clean_project_path = os.path.join(execute_path,module.split('/')[0]+'_clean')
    if os.path.exists(execute_project_path):
        shutil.rmtree(execute_project_path)
    if os.path.exists(execute_clean_project_path):
        shutil.rmtree(execute_clean_project_path)
    raw_module_path = os.path.join('dataset/raw',module.split('/')[0])
    shutil.copytree(raw_module_path,execute_clean_project_path)
    os.chdir(execute_clean_project_path)
    returncode = 1
    while returncode != 0:
        my_logger.info("running mvn clean compile, rerun if it fails")
        if('/' in module):
            pl = '/'.join(module.split('/')[1:])
            clean_result = subprocess.run(["mvn", "clean", "compile", "-pl", pl, "-Drat.skip=true", "-Dcheckstyle.skip=true", "-Dlicense.skip=true", "-Dpmd.skip=true", "-Denforcer.skip=true"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30*60)
        else:
            clean_result = subprocess.run(["mvn", "clean", "compile", "-Drat.skip=true", "-Dcheckstyle.skip=true", "-Dlicense.skip=true", "-Dpmd.skip=true", "-Denforcer.skip=true"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30*60)
        returncode = clean_result.returncode
        if(returncode!=0):
            print(clean_result.stdout,clean_result.stderr)
    my_logger.info("mvn clean compile successful")
    os.chdir('../../../')
    
    lock = multiprocessing.Manager().Lock()
    compile_pass_mutant_file = os.path.join('dataset','rawData',module,'compile_mutants.json')
    compile_mutant = []
    if os.path.exists(compile_pass_mutant_file):
        with open(compile_pass_mutant_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                compile_mutant.append(json.loads(line))
    else:
        my_logger.info('filter mutant')
        results = []
        pool2 = multiprocessing.Pool(num_processes)
        for process_id in range(num_processes):
            data_chunk = distribute_data(mutant_list, num_processes, process_id)
            result = pool2.apply_async(filter_mutant,args=(module,data_chunk,process_id,test_coverage_map,lock))
            results.append(result)
        pool2.close()
        pool2.join()
        for result in results:
            compile_mutant.extend(result.get())
    my_logger.info(f'Mutant compile filter finished. Remain mutant: {len(compile_mutant)}')
    compile_mutant = sorted(compile_mutant,
                            key=lambda x:len(test_coverage_map[x[0][14:-5].replace('/','.')+'#L'+str(x[1])]) if 'src/main/java' in x[0] 
                            else len(test_coverage_map[x[0][9:-5].replace('/','.')+'#L'+str(x[1])]),reverse=True)
    pool = multiprocessing.Pool(num_processes)
    mutant_start = 0
    for process_id in range(num_processes):
        data_chunk = distribute_data(compile_mutant[mutant_start:], num_processes, process_id)
        result = pool.apply_async(log_collect,args=(module,data_chunk,test_method_set,process_id,my_logger,lock,test_coverage_map))
    pool.close()
    pool.join()
    shutil.rmtree(execute_clean_project_path)
    return

def find_modules(project_path, module_list):
    if 'hive' in project_path:
        java_path = 'src/java'
    else:
        java_path = 'src/main/java'
    if os.path.exists(os.path.join(project_path, java_path)):
        module_list.append(project_path)
    for item in os.listdir(project_path):
        current_path = os.path.join(project_path, item)
        if os.path.isdir(current_path):
            if os.path.exists(os.path.join(current_path, java_path)):
                module_list.append(current_path)
            find_modules(current_path, module_list)

def clean_log(module,args,my_logger):
    my_logger.info('clean log data')
    process_path = os.path.join('dataset','process',module)
    clean_data_file = os.path.join('dataset/rawData', module, 'clean_log.json')
    rawLogs = []
    project_path = os.path.join('dataset','raw',module.split('/')[0])
    trace_set = set()
    module_list = []
    find_modules(project_path,module_list)
    if 'hive' in project_path:
        java_path = 'src/java'
    else:
        java_path = 'src/main/java'
    for module_path in module_list:
        file_path = os.path.join(module_path,java_path)
        trace_list = getJavaTrace(file_path,os.path.join(module_path,java_path))
        for trace in trace_list:
            trace_set.add(trace)
    trace_list = list(trace_set)
    if os.path.exists(clean_data_file):
        with open(clean_data_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                rawLogs.append(json.loads(line))
    else:
        level_list = ['ALL','TRACE','DEBUG','INFO','WARN','ERROR','FATAL']
        log_file = os.path.join('dataset/rawData', module, 'log.json')
        all_instances = []
        with open(log_file, mode="r", encoding='utf8') as f:
            for line in f:
                rawLog = json.loads(line.strip())
                testcase  = rawLog['testcase']
                label = 1 if rawLog['test']=='abnormal' else 0
                trace_log = rawLog['trace_log']
                # method_trace = rawLog['trace']
                # method_trace = '\n'.join(method_trace.split('\n')[:args.config.max_log_len])
                instance = {'testcase':testcase,'label':label,'trace_log':trace_log,'log':"",'trace':""}
                log_len = 0
                trace_len = 0
                log_flag = True
                trace_flag = True
                for line in trace_log.split('\n'):
                    if not log_flag and not trace_flag:
                        break
                    line_tokens = line.strip().split()
                    if len(line_tokens) > 2 and re.match(r'\d{4}\-\d{2}-\d{2}', line_tokens[0]) and log_flag: # log
                        if line_tokens[2] not in level_list:
                            continue
                        trace = line_tokens[3]
                        if '.'.join(trace.split('(')[0].split('.')[:-1]) not in trace_list:
                            continue
                        content = ' '.join(line_tokens[4:])
                        instance['log']+=content+'\n'
                        log_len += 1
                        if log_len >= args.config.max_log_len:
                            log_flag = False
                    elif len(line_tokens)>0 and line_tokens[0]=='#####' and trace_flag: # trace
                        instance['trace'] += line_tokens[1] + '\n'
                        trace_len += 1
                        if trace_len >= args.config.max_log_len:
                            trace_flag = False
                if log_len > 0:
                    all_instances.append(instance)
        print(f'Raw instance number:{len(all_instances)}')
        hash_dict = {}
        for instance in all_instances:
            log_hash = hash(instance['log']+instance['trace'])
            if log_hash not in hash_dict:
                hash_dict[log_hash] = [instance]
            else:
                hash_dict[log_hash].append(instance)
        clean_instance = []
        for instance in tqdm(hash_dict.values()):
            if len(set(inst['label'] for inst in instance)) == 1:
                clean_instance.extend(instance)
        with open(clean_data_file, 'w') as f:
            for instance in clean_instance:
                json.dump(instance,f)
                f.write('\n')
        print(f'Clean instance number:{len(clean_instance)}, ratio:{len(clean_instance)/len(all_instances)}')
    raw_instance = [instance for instance in clean_instance if 'ESTest' not in instance['testcase']]
    evo_instance = [instance for instance in clean_instance if 'ESTest' in instance['testcase']]
    num_raw_pos = sum([instance['label'] for instance in raw_instance])
    num_evo_pos = sum([instance['label'] for instance in evo_instance])
    num_raw = len(raw_instance)
    num_evo = len(evo_instance)
    num_total = len(clean_instance)
    num_pos = num_raw_pos + num_evo_pos
    print('Total: {} instances, {} anomaly, {} normal' \
          .format(num_total, num_pos, num_total - num_pos))
    print('Raw Test: {} instances, {} anomaly, {} normal' \
          .format(num_raw, num_raw_pos, num_raw - num_raw_pos))
    print('Evosuite Test: {} instances, {} anomaly, {} normal' \
          .format(num_evo, num_evo_pos, num_evo - num_evo_pos))

def log_parsing(log_file, config_file, log_template_file, template_mined_file, my_logger):
    if not os.path.exists(log_file):
        my_logger.error("log file does not exist. ")
        exit(-1)
    my_logger.info("Running Drain......")
    drain3(config_file, log_file, log_template_file, template_mined_file, my_logger)

def load_templates(all_line_template_file):
    with open(all_line_template_file, 'r', encoding="utf-8") as f:
        all_line_temp = []
        for template_id in f.readlines():
            all_line_temp.append(int(template_id))
    return all_line_temp

def not_empty(s):
    return s and s.strip()

def like_camel_to_tokens(camel_format):
    """
    类似驼峰命名格式转token
    可以处理类似于 addStoredBlock，StoredBlock，IOExceptionAndIOException，id，BLOCK，BLOCKException, --useDatabase, double-hummer, --reconnect-blocks
    0xADDRESS
    R63-M0-L0-U19-A
    """
    simple_format = []
    temp = ''
    flag = False

    if isinstance(camel_format, str):
        for i in range(len(camel_format)):
            if camel_format[i] == '-' or camel_format[i] == '_':
                simple_format.append(temp)
                temp = ''
                flag = False
            elif camel_format[i].isdigit():
                simple_format.append(temp)
                simple_format.append(camel_format[i])
                temp = ''
                flag = False
            elif camel_format[i].islower():
                if flag:
                    w = temp[-1]
                    temp = temp[:-1]
                    simple_format.append(temp)
                    temp = w + camel_format[i].lower()
                else:
                    temp += camel_format[i]
                flag = False
            else:
                if not flag:
                    simple_format.append(temp)
                    temp = ''
                temp += camel_format[i].lower()
                flag = True  # 需要回退
            if i == len(camel_format) - 1:
                simple_format.append(temp)
        simple_format = list(filter(not_empty, simple_format))
    return simple_format

def get_pure_templates(pure_template_path, drain_template_list_file, my_logger):
    templates = []
    stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
                 "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's",
                 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs',
                 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is',
                 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
                 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
                 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
                 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
                 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
                 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've",
                 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
                 "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
                 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn',
                 "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    if os.path.exists(drain_template_list_file):
        my_logger.info("Read from logTemplates.txt")
        with open(drain_template_list_file, 'r', encoding='utf-8') as reader:
            for line in reader.readlines():
                if line != "":
                    # Get pure templates.
                    pure_line = re.sub(r'[^\w\d\/\_]+', ' ', line.strip())
                    pure_line_token = pure_line.split()
                    for i in range(len(pure_line_token)):
                        if pure_line_token[i][0] == '/':
                            pure_line_token[i] = ""
                        else:
                            if bool(re.search(r'\d', pure_line_token[i])):
                                pure_line_token[i] = ""
                            elif len(pure_line_token[i]) == 1:
                                pure_line_token[i] = ""
                            else:
                                simple_words = like_camel_to_tokens(pure_line_token[i])
                                pure_line_token[i] = " ".join(simple_words)
                                pure_line_token[i] = pure_line_token[i].lower()
                                    
                    pure_line_token = list(filter(not_empty, pure_line_token))  # 除去 ''
                    pure_line_token = list(filter(lambda x: x.lower() not in stopwords, pure_line_token))
                    l = ' '.join(pure_line_token)
                    if len(l) > 0:
                        templates.append(l)
                    else:
                        templates.append("this_is_an_empty_event")
                else:
                    templates.append("this_is_an_empty_event")
    else:
        my_logger.error("No drain_template_list_file %s" % drain_template_list_file)
        exit(-1)
    if os.path.exists(pure_template_path):
        os.remove(pure_template_path)
    with open(pure_template_path, 'w', encoding="utf-8") as writer:
        for template in templates:
            writer.write(template + '\n')
    my_logger.info("Save pure template %s." % pure_template_path)

def get_unduplicated_templates(pure_template_file,all_line_template_file,clean_template_file,my_logger):
    my_logger.info(f'remove duplicate templates and constrcut map between original and new templates id')
    with open(pure_template_file, 'r', encoding='utf-8') as reader:
        templates = reader.readlines()
    clean_id = 0
    clean_templates = []
    hash_map = []
    for template in templates:
        if template.strip() not in clean_templates:
            hash_map.append(clean_id)
            clean_templates.append(template.strip())
            clean_id+=1
        else:
            hash_map.append(clean_templates.index(template.strip()))
    with open(clean_template_file, 'w', encoding='utf-8') as f:
        for template in clean_templates:
            f.write(template+'\n')
    with open(all_line_template_file, 'r', encoding='utf-8') as reader:
        line_template = reader.readlines()
    clean_line_template = [hash_map[int(template.strip())] for template in line_template]
    with open(all_line_template_file, 'w', encoding='utf-8') as f:
        for template in clean_line_template:
            f.write(str(template)+'\n')


def count_testmethod_withlog(module, my_logger):
    my_logger.info('Count the test method number with log.')
    count_file = os.path.join('dataset','rawData',module,'method_count.txt')
    if os.path.exists(count_file):
        with open(count_file, 'r') as f:
            method_log_count = int(f.read().strip())
    else:
        method_file = os.path.join('dataset','rawData',module,'testMethod.txt')
        test_method_set = set()
        with open(method_file,'r') as f:
            for line in f.readlines():
                if line.strip()!="":
                    test_method_set.add(line.strip())
        try:
            level_list = ['ALL','TRACE','DEBUG','INFO','WARN','ERROR','FATAL']
            execute_path = os.path.join('dataset','execute')
            if not os.path.exists(execute_path):
                os.makedirs(execute_path)
            execute_project_path = os.path.join(execute_path,module.split('/')[0]+'_count')
            raw_module_path = os.path.join('dataset/raw',module.split('/')[0])
            if os.path.exists(execute_project_path):
                shutil.rmtree(execute_project_path)
            shutil.copytree(raw_module_path,execute_project_path)
            method_log_count = 0
            if('/' in module):
                pl = '/'.join(module.split('/')[1:])
                os.chdir(execute_project_path)
                # check if file is illegal
                with tqdm(total=len(list(test_method_set))) as pbar:
                    for test_method in list(test_method_set):
                        try:
                            test_class = test_method.split('#')[0]
                            with open('test.log','w') as f:
                                process = subprocess.run(["mvn", "clean", "test", "-Dtest="+test_method, "-pl", pl ,"-Drat.skip=true", "-Dcheckstyle.skip=true", "-Dlicense.skip=true", "-Dpmd.skip=true", "-Denforcer.skip=true"], stdout=f, stderr=subprocess.PIPE)
                            log_content = ""
                            trace = ""
                            with open('test.log','r') as log_file:
                                for line in log_file:
                                    line_tokens = line.strip().split()
                                    if len(line_tokens)>0 and line_tokens[0]=='#####':
                                        trace+=line_tokens[1]+'\n'
                                    elif len(line_tokens) > 2 and re.match(r'\d{4}\-\d{2}-\d{2}', line_tokens[0]):
                                        if line_tokens[2] not in level_list:
                                            continue
                                        log_content += line
                            if len(log_content)==0:
                                test_method_set.discard(test_method)
                            else:
                                method_log_count += 1
                        except Exception as e:
                            traceback.print_exc()
                        pbar.update(1)
                os.chdir('../../../')
            else:
                os.chdir(execute_project_path)
                # check if file is illegal
                with tqdm(total=len(list(test_method_set))) as pbar:
                    for test_method in tqdm(list(test_method_set)):
                        try:
                            test_class = test_method.split('#')[0]
                            with open('test.log','w') as f:
                                process = subprocess.run(["mvn", "clean", "test", "-Dtest="+test_method, "-Drat.skip=true", "-Dcheckstyle.skip=true", "-Dlicense.skip=true", "-Dpmd.skip=true", "-Denforcer.skip=true"], stdout=f, stderr=subprocess.PIPE)
                            log_content = ""
                            trace = ""
                            with open('test.log','r') as log_file:
                                for line in log_file:
                                    line_tokens = line.strip().split()
                                    if len(line_tokens)>0 and line_tokens[0]=='#####':
                                        trace+=line_tokens[1]+'\n'
                                    elif len(line_tokens) > 2 and re.match(r'\d{4}\-\d{2}-\d{2}', line_tokens[0]):
                                        if line_tokens[2] not in level_list:
                                            continue
                                        trace = line_tokens[3]
                                        log_content += line
                            if len(log_content)==0:
                                test_method_set.discard(test_method)
                            else:
                                method_log_count += 1
                        except Exception as e:
                            traceback.print_exc()
                        pbar.update(1)
                os.chdir('../../../')
            with open(method_file, 'w') as f:
                for test_method in test_method_set:
                    f.write(test_method+'\n')
            shutil.rmtree(execute_project_path,ignore_errors=True)
            with open(count_file, 'w') as f:
                f.write(str(method_log_count))
        except Exception as e:
            traceback.print_exc()
            exit(-1)
    my_logger.info(f'test method number with log: {method_log_count}')
    if method_log_count < 50:
        exit(-1)
    else:
        return