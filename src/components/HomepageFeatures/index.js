import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: 'Python Programming',
    Svg: require('@site/static/img/logo.svg').default,
    description: (
      <>
        Master Python programming fundamentals and advanced concepts specifically
        tailored for quantitative finance. Learn NumPy, Pandas, and data visualization.
      </>
    ),
  },
  {
    title: 'Financial Markets',
    Svg: require('@site/static/img/logo.svg').default,
    description: (
      <>
        Understand market structures, trading mechanisms, and financial instruments.
        Learn about stocks, options, futures, and derivatives.
      </>
    ),
  },
  {
    title: 'Quantitative Strategies',
    Svg: require('@site/static/img/logo.svg').default,
    description: (
      <>
        Develop and backtest trading strategies using statistical methods and
        machine learning. Master portfolio optimization and risk management.
      </>
    ),
  },
];

function Feature({Svg, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
