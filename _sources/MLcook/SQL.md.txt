# SQL

## 1. 基础过滤与排序
* **空值处理**：字符串`column != 'null'` , 空值`column IS NOT NULL`。
* where 不能对select的别名进行过滤，必须重复写表达式或者使用with。
* **模糊匹配**：`where university like '%北京%'`。
* **集合过滤**：`where university in ('北京大学', '复旦大学', '山东大学')`。
* **多列排序**：`order by gpa desc, age desc`（先按 GPA 降序，相同则按年龄降序）。`order by device_id` 默认升序。

## 2. 连接，联合
* 横向连接（JOIN）：
    ```sql
    select
        user_profile.university,
        qd.difficult_level,
        count(qpd.device_id) / count(distinct user_profile.device_id)
    from
        user_profile
        inner join question_practice_detail as qpd
        on qpd.device_id = user_profile.device_id
        inner join question_detail as qd
        on qd.question_id = qpd.question_id
    where user_profile.university = '山东大学'
    group by qd.difficult_level
    ```
* 纵向连接（UNION ALL）：
    ``` sql
    select
        device_id, gender,age,gpa
    from user_profile
    where university = '山东大学'
    union all
    select
        device_id, gender,age,gpa
    from user_profile
    where gender = 'male'
    ```



## 2. 聚合与分组 (Aggregation)
* **核心规则**：`WHERE` 过滤行，`HAVING` 过滤组。
* **常见组合**：
    ```sql
    select university, avg(gpa) as avg_gpa
    from user_profile
    group by university
    having avg_gpa > 3.5;
    ```
* **计数技巧**：`count(column)` 忽略 NULL，`count(*)` 统计所有行。
    `count(qpd.question_id) as total_cnt, `   none会统计为0
* 其他聚合函数：`SUM()`, `AVG()`, `MAX()`, `MIN()`,  等。会忽略null

## 3. 条件逻辑 (Conditional)
* **简单条件**：`if(age >= 25, '25岁及以上', '25岁以下')`。   条件求和 `sum(if(qpd.result = 'right', 1, 0)) as right_question_cnt`
* **复杂分支 (CASE WHEN)**：
    ```sql
    case
        when age < 20 then '20岁以下'
        when age between 20 and 24 then '20-24岁'
        else '25岁及以上'
    end as age_cut
    ```

## 4. 字符串与日期处理
* **字符串切割**：`substring_index(profile, ',', -1)`（提取最后一个逗号后的内容）。
* **正则匹配**：`phone_number REGEXP '^[1-9][0-9]{2}-?[0-9]{3}-?[0-9]{4}$'`。
* **字符串拼接**：`concat(round(count(overdue_days)/count(*) * 100, 1), '%')`
* **日期处理**：
    * `year(date) = 2021 and month(date) = 8`
    * `date_format(t_time, '%Y%m')`
    * `datediff(end, start)` 计算天数差。
* 取整差值：`TIMESTAMPDIFF(UNIT, start, end)` (UNIT 可以是 DAY, HOUR, MINUTE, SECOND)。精确差值：算出秒数 SECOND 再除以 60 或 3600。
* `DATE_SUB(log_date, INTERVAL rn DAY) as base_date` 以 log_date 为基准，向前推 rn 天。

## 5. 窗口函数 (Window Functions)

* **语法**：`ROW_NUMBER() OVER(PARTITION BY 分组列 ORDER BY 排序列)`。
* **区别**：
    * `ROW_NUMBER()`：连续排名（1, 2, 3）。
    * `RANK()`：跳跃排名（1, 1, 3）。会重复排名
    * `DENSE_RANK()`：连续重复排名（1, 1, 2）。
    ``` sql
    SELECT
        device_id,
        university,
        gpa,
        ROW_NUMBER() OVER(PARTITION BY university ORDER BY gpa ASC) AS rn
    FROM user_profile
    ```

## 6. 高级进阶
* **次日留存率逻辑**：
    通过 `LEFT JOIN` 同一张表，连接条件为 `q1.device_id = q2.device_id AND q2.date = DATE_ADD(q1.date, INTERVAL 1 DAY)`，最后用 `count(q2.date)/count(q1.date)` 计算。
* **CTE (Common Table Expressions)**：使用 `with t1 as (...)` 让逻辑更清晰。
* **行转列 (Unpivoting)**：通过 `UNION ALL` 将宽表（多个课程列）转化为长表（一列课程名，一列值）。
    ```sql
    
    union all 
    (
        select
        'course1' as course_name, course1_cnt as cnt
        from t1

        union all

        select
        'course2' as course_name, course2_cnt as cnt
        from t1
        union all

        select
        'course3' as course_name, course3_cnt as cnt
        from t1
    )```


## 7. SQL 执行顺序 (面试高频)
这是理解 SQL 逻辑的**金钥匙**，请务必记住：
执行顺序”而非“书写顺序”

1.  **FROM / JOIN** (加载数据源)
2.  **ON** (连接条件)
3.  **WHERE** (初步过滤：过滤掉不要的行)
4.  **GROUP BY** (分组：把剩下的行分组)
5.  **聚合函数** (COUNT, SUM等。算出每组的值)
6.  **HAVING** (组过滤：过滤掉不符合条件的组)
7.  **WINDOW FUNCTIONS** (计算排名)
8.  **SELECT** (确定输出列)
9.  **DISTINCT** (去重)
10. **ORDER BY** (排序)
11. **LIMIT** (截取结果)


## 8. with
 ```sql
with
t1 as (查询)
t2 as (。。。）
select , from t1,t2
 ```

## 10. 常见题目
* 连续登录天数
```sql
WITH t1 AS (
    -- 1. 先把时间转为日期，并去重（防止一天多次登录干扰计算）
    SELECT DISTINCT user_id, DATE(log_time) as log_date
    FROM log_table
),
t2 AS (
    -- 2. 给每个用户的登录日期排个名
    SELECT 
        user_id, 
        log_date,
        ROW_NUMBER() OVER(PARTITION BY user_id ORDER BY log_date) as rn
    FROM t1
),
t3 AS (
    -- 3. 日期减去排名，得到一个“起始基准日”
    -- 如果日期连续，这个 base_date 会完全一样
    SELECT 
        user_id,
        DATE_SUB(log_date, INTERVAL rn DAY) as base_date
    FROM t2
)
-- 4. 统计每个基准日出现的次数，即为连续登录天数
SELECT 
    user_id, 
    COUNT(*) as consecutive_days
FROM t3
GROUP BY user_id, base_date
HAVING consecutive_days >= 3; -- 比如过滤出连续登录3天以上的

```














FLOOR向下


DISTINCT user_id, DATE(log_time)
对后面所有字段管用

count(distinct ut.user_id), 
去重后计数


select
ut.user_id, ut.vip, 
if (ot.order_price IS NULL, 0, ot.order_price) as price
from uservip_tb as ut
left join order_tb as ot
on ot.user_id = ut.user_id
处理没有订单的用户，订单金额记为0



select
    user_id, count(*) as visit_nums
from
    visit_tb
where
    user_id in (
        select distinct
            user_id
        from
            order_tb
        WHERE
            DATE(order_time) = '2022-09-02'
    )
    and DATE(visit_time) = '2022-09-02'
group by user_id


SELECT
user_id,min(log_time)
FROM login_tb
group by DATE(log_time), user_id
每天最早记录


    from t2
    where music_id NOT in (
        select music_id from t3
    )

    HAVING watch_cnt > 5




            # 次数，视频时间排名
    select
    cid,  sum(watch_cnt) as pv,
    row_number() over(order by sum(watch_cnt) desc, min(release_date) desc) as rk
    from t2
    group by cid

    先根据cid分组了。 然后计算出现的聚合函数值。
    得到  cid - sum_watch - min_relea
    
    

    WHERE NOT (in_time > '12:00:00' AND out_time < '11:00:00')
in_time NOT BETWEEN '11:00:00' AND '12:00:00'