use std::marker::PhantomData;

use halo2_proofs::{
    arithmetic::Field,
    circuit::{layouter::TableLayouter, Cell, Chip, Layouter, Value, Region, AssignedCell},
    dev::{MockProver, VerifyFailure},
    pasta::Fp,
    plonk::{
        Advice, Assignment, Circuit, Column, ConstraintSystem, Error, Expression, Fixed,
        Selector
    },
    poly::Rotation,
};

// 定义卷积芯片
struct ConvChip<F: Field> {
    config: ConvChipConfig,
    _marker: PhantomData<F>,
}

// 定义卷积芯片配置
#[derive(Clone, Debug)]
struct ConvChipConfig {
    input: Column<Advice>,
    output: Column<Advice>, //存储输出的值
    kernel: Column<Advice>, // 定义一个列kernel，用于存储kernel参数
    bias: Column<Advice>, // 定义一个列bias，用于存储bias参数
}


// 实现卷积芯片配置
impl ConvChipConfig {
    // 配置卷积芯片
    fn configure(meta: &mut ConstraintSystem<Fp>) -> Self {
        // 定义四个列
        let input = meta.advice_column();
        let output = meta.advice_column();
        let kernel = meta.advice_column(); // 定义一个列kernel，用于存储kernel参数
        let bias = meta.advice_column(); // 定义一个列bias，用于存储bias参数
        
        // 定义一个约束，要求output[i] = input[i] * kernel[i] + bias[i]，其中kernel[i]和bias[i]是列kernel和列bias中的值
        meta.create_gate("output[i] = input[i] * kernel[i] + bias[i]", |meta| {
            let expressions = vec![
                meta.query_advice(input, Rotation::cur()),
                meta.query_advice(output, Rotation::cur()),
                meta.query_advice(kernel, Rotation::cur()),
                meta.query_advice(bias, Rotation::cur()),
            ];
            // 将向量转换成一个迭代器
            let mut iter = expressions.into_iter();
            // 获取迭代器中的值
            let input = iter.next().unwrap();
            let output = iter.next().unwrap();
            let kernel = iter.next().unwrap();
            let bias = iter.next().unwrap();
            // 计算output[i] - input[i] * kernel[i] - bias[i]
            vec![output - (input * kernel) - bias]
        });
        // 返回配置
        Self { input, output, kernel, bias }
    }
}

// 实现卷积芯片
impl<F: Field> Chip<F> for ConvChip<F> {
    type Config = ConvChipConfig;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}

impl<F: Field> ConvChip<F> {
    // 构造卷积芯片
    fn construct(config: <Self as Chip<F>>::Config) -> Self {
        Self { 
            config,
            _marker: PhantomData,
        }
    }
    
    // 调用卷积芯片
    fn conv(
        &self,
        mut layouter: impl Layouter<F>,
        input: Column<Advice>,
        input_len: usize, // 输入数据的长度
    ) -> Result<Column<Advice>, Error> {
        // 分配一个行，用于存储output
        layouter.assign_region(
            || "conv",
            |mut region| {
                // 获取配置
                let config = self.config.clone();
                // 获取input的值，假设input是一个n*n长度的向量，表示一个n*n的矩阵
                let n = (input_len as f64).sqrt() as usize;
                let mut input_val = vec![];
                for i in 0..n*n {
                    input_val.push(region.query_advice(config.input, i));
                }
                // 获取kernel和bias的值，假设kernel和bias是两个n*n长度的向量，表示两个n*n的矩阵
                let mut kernel_val = vec![];
                let mut bias_val = vec![];
                for i in 0..n*n {
                    kernel_val.push(region.query_advice(config.kernel, i));
                    bias_val.push(region.query_advice(config.bias, i));
                }
                // 计算output的值
                let mut output_val = vec![];
                for i in 0..n*n {
                    // 计算output[i] = input[i] * kernel[i] + bias[i]
                    output_val.push((input_val[i] * kernel_val[i]) + bias_val[i]);
                    // 分配output[i]的值到列output中
                    region.assign_advice(|| format!("assign output[{}]", i), config.output, i, || Ok(output_val[i]))?;
                }
                // 返回列output
                Ok(config.output)
            },
        )
    }
}
    
